import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))

import wandb
import argparse
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model
from utils.tools import parse_config_file, same_seeds, save_metrix, get_exp_name
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
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--T', default=1., type=float)
    parser.add_argument('--start_kd', default=0, type=int)
    return parser.parse_args()

# Parase config file and initiate logging
configs = parse_config_file(parse_args())

configs.ADV.KD.beta = configs.beta
configs.ADV.KD.T = configs.T
configs.ADV.KD.start_kd = configs.start_kd
configs.exp_name = get_exp_name(configs.method, configs.dataset, configs)

wandb.init(
    # set the wandb project where this run will be logged
    project="TransferAttack_LAFA",
    # track hyperparameters and run metadata
    config=configs,
    name=configs.exp_name
)


def get_avg_metrix(surrogate_model, metrix):
    result = 0.
    num = 0
    for model_name, metrix_one in metrix.items():
        if model_name == surrogate_model:
            continue
        result += metrix_one.attack_rate
        num += 1
    return result / num


def convert_metrix2dict(metrix):
    results = {}
    for model_name, metrix_one in metrix.items():
        results[model_name] = metrix_one.attack_rate
    return results


def main():
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    print(f'surrogate model: {configs.model}\tattack method: {configs.method}')

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
    result = convert_metrix2dict(metrix)
    result['avg.'] = get_avg_metrix(configs.model, metrix)
    wandb.log(
        data=result
    )


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

