import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))

import argparse
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model, get_model_feature_only
from utils.tools import parse_config_file, same_seeds, save_metrix
from sklearn.manifold import TSNE
import warnings
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


def main():
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)
    models_feat = get_model_feature_only(configs, device=device)
    loss_func = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()

    print(f'ens model: {configs.model}\tattack method: {configs.method}')

    for idx, (data, label) in enumerate(tqdm(test_loader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)

        # get attack
        attack_method = get_attack(configs, model=models[configs.model], device=device)
        adv_exp = attack_method(data, label)

        dis = {}
        feature_tsne = {}
        for model_name, model_feat in models_feat.items():
            model = models[model_name]
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
                feat_clean = model_feat(data)
                feat_adv = model_feat(adv_exp)
            # Logits
            # loss_clean = loss_func(r_clean, label)
            # loss_adv = loss_func(r_adv, label)
            loss_clean = loss_mse(r_clean, torch.nn.functional.one_hot(label, num_classes=1000))
            loss_adv = loss_mse(r_adv, torch.nn.functional.one_hot(label, num_classes=1000))
            # feat distance
            dis[model_name] = []
            for i in range(len(feat_clean)):
                # d = torch.mean(torch.abs(feat_adv[i] - feat_clean[i]))
                d = loss_mse(feat_adv[i], feat_clean[i])
                if len(dis[model_name]) != len(feat_clean):
                    dis[model_name].append(d.item())
                else:
                    dis[model_name][i] += d.item()
            if len(dis[model_name]) != len(feat_clean) + 2:
                dis[model_name].append(loss_clean.item())
                dis[model_name].append(loss_adv.item())
            else:
                dis[model_name][-2] += loss_clean.item()
                dis[model_name][-1] += loss_adv.item()

    save_metrix(configs, metrix)


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

