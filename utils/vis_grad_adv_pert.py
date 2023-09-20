import os
import sys

import torchvision.utils

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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
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

save_path = './vis'

def main():
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    attack_methods = ['fgsm', 'pgd', 'mi-fgsm', 'di-fgsm', 'pi-fgsm', 'ti-fgsm', 'ni-fgsm', 'ni-fgsm', 'sini-fgsm',
                      'vmi-fgsm', 'vni-fgsm', 'smi-fgsm', 'lafa', 'lafa_ni', 'lafa_ti', 'lafa_di',
                      'lafa_vmi']

    b = configs.DATA.batch_size

    cam_model = models['resnet152']
    target_layer = [cam_model[-1].layer4[-1]]

    for method in attack_methods:
        configs.method = method
        print(f'surrogate model: {configs.model}\tattack method: {configs.method}')

        for idx, (data, label) in enumerate(tqdm(test_loader)):
            n = label.size(0)
            data, label = data.to(device), label.to(device)
            # get attack
            attack_method = get_attack(configs, model=models[configs.model], device=device)
            adv_exp = attack_method(data, label)
            pert = adv_exp - data

            cam_vis = []
            for i in range(b):
                one_adv_data, one_label = adv_exp[i:i+1], label[i:i+1]
                cam = GradCAM(model=cam_model, target_layers=target_layer, use_cuda=True)
                targets = [ClassifierOutputTarget(one_label.item())]
                grayscale_cam = cam(input_tensor=one_adv_data, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                one_adv_data_np = one_adv_data[0].permute(1, 2, 0).cpu().detach().numpy()
                visualization = show_cam_on_image(one_adv_data_np, grayscale_cam, use_rgb=True)
                cam_vis.append(torch.from_numpy(visualization / 255.).permute(2, 0, 1))
            cam_vis = torch.stack(cam_vis, dim=0)

            for i in range(b):
                l = label[i].item()
                cur_path = os.path.join(save_path, str(l))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)

                torchvision.utils.save_image(data[i:i+1], os.path.join(cur_path, 'data.png'),
                                             normalize=True)
                torchvision.utils.save_image(adv_exp[i:i+1], os.path.join(cur_path, f'adv_{method}.png'),
                                             normalize=True)
                torchvision.utils.save_image(pert[i:i+1], os.path.join(cur_path, f'pert_{method}.png'),
                                             normalize=True)
                torchvision.utils.save_image(cam_vis[i:i + 1], os.path.join(cur_path, f'cam_{method}.png'),
                                             normalize=True)


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

