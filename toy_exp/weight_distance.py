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

    data = {'conv': {}, 'layer1': {}, 'layer2': {}, 'layer3': {}, 'layer4': {}}

    for feat_name, feat_weight in models_feat.items():
        data['conv'][feat_name] = feat_weight[-1].conv1.weight
        data['layer1'][feat_name] = feat_weight[-1].layer1[-1].conv1.weight
        data['layer2'][feat_name] = feat_weight[-1].layer2[-1].conv1.weight
        data['layer3'][feat_name] = feat_weight[-1].layer3[-1].conv1.weight
        data['layer4'][feat_name] = feat_weight[-1].layer4[-1].conv1.weight

    print('stop')


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

