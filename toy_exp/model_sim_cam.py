import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))

import argparse
import torch
import torchvision
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from model.get_models import get_models, get_teacher_model
from utils.tools import parse_config_file, same_seeds, save_metrix
import warnings

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
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


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def main():
    configs.DATA.batch_size = 1
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    print(f'ens model: {configs.model}\tattack method: {configs.method}')
    path = './vis_cam'
    for idx, (data, label) in enumerate(tqdm(test_loader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        images = []
        for model_name, model in models.items():
            if 'wrn' in model_name:
                target_layer = [model[-1].conv1]
                # target_layer = [model[-1].layer4[-1]]
            elif 'resnet' in model_name:
                target_layer = [model[-1].conv1]
                # target_layer = [model[-1].layer4[-1]]
            elif 'inc_v3' in model_name:
                target_layer = [model[-1].Conv2d_1a_3x3, model[-1].Conv2d_2a_3x3, model[-1].Conv2d_2b_3x3]
                # target_layer = [model[-1].Mixed_7b, model[-1].Mixed_7c]
            elif 'inc_v4' in model_name:
                target_layer = [model[-1].features[0], model[-1].features[1], model[-1].features[2]]
                # target_layer = [model[-1].features[-1], model[-1].features[-2], model[-1].features[-3]]
            elif 'inc_resv2' in model_name:
                target_layer = [model[-1].conv2d_1a, model[-1].conv2d_2a, model[-1].conv2d_2b]
                # target_layer = [model[-1].conv2d_7b]
            elif 'bit' in model_name:
                target_layer = [model[-1].stem[0]]
                # target_layer = [model[-1].stages[-1].blocks[-1]]
            elif 'dense' in model_name:
                target_layer = [model[-1].features.conv0]
                # target_layer = [model[-1].features[-2]]
            elif 'vit' in model_name or 'deit' in model_name:
                target_layer = [model[-1].blocks[0].norm1]
                # target_layer = [model[-1].blocks[-1].norm1]
            elif 'swin' in model_name:
                target_layer = [model[-1].layers[-1].blocks[-1].norm1]
                # target_layer = [model[-1].layers[-1].blocks[-1].norm1]
            elif model_name == 'incv3_ens3' or model_name == 'incv3_ens4':
                target_layer = [model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3]
                # target_layer = [model.Mixed_7b, model.Mixed_7c]
            elif model_name == 'incv2_ens':
                target_layer = [model.conv2d_1a, model.conv2d_2a, model.conv2d_2b]
                # target_layer = [model.conv2d_7b]
            else:
                raise NotImplemented

            if 'vit' in model_name or 'deit' in model_name:
                cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True,
                              reshape_transform=reshape_transform_vit)
            elif 'swin' in model_name:
                cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True,
                              reshape_transform=reshape_transform_swin)
            else:
                cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
            targets = [ClassifierOutputTarget(label.item())]
            grayscale_cam = cam(input_tensor=data, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(data[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam, use_rgb=True)
            images.append(torch.from_numpy(visualization / 255.).permute(2,0,1).unsqueeze(0))
        images_grid = torchvision.utils.make_grid(torch.concat(images), padding=2, nrow=5)

        cur_path = os.path.join(path, str(label.item()))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        torchvision.utils.save_image(images_grid, os.path.join(cur_path, 'first_layer.png'))
        # torchvision.utils.save_image(images_grid, os.path.join(cur_path, 'last_layer.png'))













if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()

