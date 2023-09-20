import os
import sys

import torchvision.utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'classifiers')))

import argparse
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model, get_model_feature_only
from utils.tools import parse_config_file, same_seeds, save_metrix
import warnings
from sklearn import decomposition
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='training dataset, e.g. cifar10/100 or imagenet')
    parser.add_argument('--model', required=True, help='the model for black-box attack')
    parser.add_argument('-m', '--method', default='standard', type=str,
                        help='at method, e.g. standard at, trades, mart, etc.')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu id used for training.')
    return parser.parse_args()

# Parase config file and initiate logging
configs = parse_config_file(parse_args())

MAX_ITERS = 30

attack_methods = ['fgsm', 'pgd', 'mi-fgsm', 'di-fgsm', 'pi-fgsm', 'ti-fgsm', 'ni-fgsm', 'sini-fgsm', 'vmi-fgsm', 'vni-fgsm', 'Admix']
# attack_methods = ['fgsm', 'pgd']


def main():
    configs.DATA.batch_size = 10
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models = get_model_feature_only(configs, device=device)
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    pca = decomposition.PCA(3)

    results = []
    n = 0
    for idx, (data, label) in enumerate(tqdm(test_loader)):
        data, label = data.to(device), label.to(device)
        feats = {}
        for model_name, model in models.items():
            with torch.no_grad():
                feat = model(data)
            for i in range(len(feat)):
                feat[i] = feat[i].cpu().detach()
            feats[model_name] = feat
        loss_mse = torch.nn.MSELoss()
        target = feats['inc_v3']
        for model_name, model in feats.items():
            result = []
            for j in range(5):
                # distance = loss_mse(target[j], feats[model_name][j])
                t = target[j]
                f = feats[model_name][j]
                t = t.flatten(1)
                f = f.flatten(1)
                # t_pca = pca.fit_transform(t)
                # f_pca = pca.fit_transform(f)
                t_pca = torch.pca_lowrank(t, q=10)[0]
                f_pca = torch.pca_lowrank(f, q=10)[0]
                cos_result = torch.nn.CosineSimilarity()(t_pca, f_pca).mean()
                result.append(cos_result.item())
            if len(results) == 0:
                results = result
            else:
                for i in range(len(result)):
                    results[i] += result[i]
            n += 1
    for i in range(len(results)):
        results[i] = results[i] / n

    print(results)

if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()





