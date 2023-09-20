import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))

import argparse
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model
from utils.tools import parse_config_file, same_seeds, save_metrix
import warnings
from torchstat import stat
import copy
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


def prune(model, model_name):
    # from torch.nn.utils import prune
    # from collections import OrderedDict
    #
    # for name, module in model.named_parameters():
    #     p = prune.L1Unstructured(module)
    #     p.
    #
    # print(f"{model_name} have {sum(x.numel() for x in model.parameters())} paramerters in total")
    pass


def main():
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    # stat(copy.deepcopy(models[configs.model]).cpu(), (3,224,224))
    prune(model=models[configs.model][-1], model_name=configs.model)

    print(f'ens model: {configs.model}\tattack method: {configs.method}')

    for idx, (data, label) in enumerate(tqdm(test_loader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        # get attack
        attack_method = get_attack(configs, model=models[configs.model], device=device)
        adv_exp = attack_method(data, label)

        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()

            metrix[model_name].update(correct_clean, correct_adv, n)

    save_metrix(configs, metrix)


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

