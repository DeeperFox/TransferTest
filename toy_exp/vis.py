import os
import sys

import timm
import torchvision.utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'classifiers')))

import argparse
import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model
from utils.tools import parse_config_file, same_seeds, save_metrix
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

MAX_ITERS = 30

attack_methods = ['fgsm', 'pgd', 'mi-fgsm', 'di-fgsm', 'pi-fgsm', 'ti-fgsm', 'ni-fgsm', 'sini-fgsm', 'vmi-fgsm',
                  'vni-fgsm', 'Admix', 'kd-fgsm']
attack_methods_at = ['fgsm', 'pgd', 'mi-fgsm']


def main():
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    model_at = timm.create_model('resnet18',
                                 checkpoint_path='../checkpoints/pretrain_at/imagenet/resnet18.pth')
    model_at.eval()
    model_at.to(device)

    results = {}
    results_at = {}

    for one_method in attack_methods:
        configs.method = one_method
        results[configs.method] = {}
        print(f'ens model: {configs.model}\tattack method: {configs.method}')
        for idx, (data, label) in enumerate(tqdm(test_loader)):
            n = label.size(0)
            data, label = data.to(device), label.to(device)
            # get attack
            attack_method = get_attack(configs, model=models[configs.model], device=device)
            adv_exp = attack_method(data, label)
            pert = adv_exp - data
            pert_norm = (pert - pert.min()) / (pert.max() - pert.min())
            for i in range(n):
                results[configs.method][f'{label[i].item()}'] = {'image': data[i:i+1].cpu().detach(),
                                                                 'adv_img': adv_exp[i:i+1].cpu().detach(),
                                                                 'pert': pert_norm[i:i+1].cpu().detach()}

    for one_method in attack_methods_at:
        configs.method = one_method
        results_at[configs.method] = {}
        print(f'ens model: {configs.model}\tattack method: {configs.method}')
        for idx, (data, label) in enumerate(tqdm(test_loader)):
            n = label.size(0)
            data, label = data.to(device), label.to(device)
            # get attack
            attack_method = get_attack(configs, model=model_at, device=device)
            adv_exp = attack_method(data, label)
            pert = adv_exp - data
            pert_norm = (pert - pert.min()) / (pert.max() - pert.min())
            for i in range(n):
                results_at[configs.method][f'{label[i].item()}'] = {'image': data[i:i + 1].cpu().detach(),
                                                                    'adv_img': adv_exp[i:i + 1].cpu().detach(),
                                                                    'pert': pert_norm[i:i + 1].cpu().detach()}

    path = './vis_resnet18'
    for i in range(1000):
        result = None
        for one_method in attack_methods:
            if result is None:
                result = torch.concat([results[one_method][str(i)]['image'],
                                       results[one_method][str(i)]['adv_img'],
                                       results[one_method][str(i)]['pert']])
            else:
                result = torch.concat([result, results[one_method][str(i)]['image'],
                                       results[one_method][str(i)]['adv_img'],
                                       results[one_method][str(i)]['pert']])
        results_img = torchvision.utils.make_grid(result, padding=5, nrow=3)
        cur_path = os.path.join(path, str(i))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        torchvision.utils.save_image(results_img, os.path.join(cur_path, f'{i}.png'))

        for idx, one_method in enumerate(attack_methods):
            if idx == 0:
                torchvision.utils.save_image(results[one_method][str(i)]['image'],
                                             os.path.join(cur_path, f'image.png'))
            torchvision.utils.save_image(results[one_method][str(i)][f'adv_img'],
                                         os.path.join(cur_path, f'adv_{one_method}.png'))
            torchvision.utils.save_image(results[one_method][str(i)][f'pert'],
                                         os.path.join(cur_path, f'pert_{one_method}.png'))

    for i in range(1000):
        result = None
        for one_method in attack_methods_at:
            if result is None:
                result = torch.concat([results_at[one_method][str(i)]['image'],
                                       results_at[one_method][str(i)]['adv_img'],
                                       results_at[one_method][str(i)]['pert']])
            else:
                result = torch.concat([result, results_at[one_method][str(i)]['image'],
                                       results_at[one_method][str(i)]['adv_img'],
                                       results_at[one_method][str(i)]['pert']])
        results_img = torchvision.utils.make_grid(result, padding=5, nrow=3)
        cur_path = os.path.join(path, str(i))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        torchvision.utils.save_image(results_img, os.path.join(cur_path, f'{i}_adv.png'))

        for idx, one_method in enumerate(attack_methods_at):
            torchvision.utils.save_image(results_at[one_method][str(i)][f'adv_img'],
                                         os.path.join(cur_path, f'adv_{one_method}_ResNet18_at.png'))
            torchvision.utils.save_image(results_at[one_method][str(i)][f'pert'],
                                         os.path.join(cur_path, f'pert_{one_method}_ResNet18_at.png'))

        print(i)




if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()



