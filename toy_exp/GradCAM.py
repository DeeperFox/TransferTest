import torchvision.utils
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    configs.DATA.batch_size = 1
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    # get models
    models, metrix = get_models(configs, device=device)

    test_models = ['wrn50_2', 'wrn101_2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    for model_name in test_models:
        print(model_name)
        cur_model = models[model_name]
        # target_layers = [cur_model[-1].conv1]
        target_layers = [cur_model[-1].layer3[-1].conv1]
        cam = GradCAM(model=cur_model, target_layers=target_layers, use_cuda=True)
        for idx, (data, label) in enumerate(tqdm(test_loader)):
            data, label = data.to(device), label.to(device)
            targets = [ClassifierOutputTarget(label.item())]
            gradscale_cam = cam(input_tensor=data, targets=targets)
            gradscale_cam = gradscale_cam[0, :]
            visualization = show_cam_on_image(data[0].permute(1, 2, 0).detach().cpu().numpy(), gradscale_cam,
                                              use_rgb=True)
            visualization_torch = torch.from_numpy(visualization / 255.)

            # save
            cur_path = os.path.join('./vis_cam', str(label.item()))
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)
            torchvision.utils.save_image(data, os.path.join(cur_path, 'data.png'))
            # torchvision.utils.save_image(visualization_torch.permute(2, 0, 1).unsqueeze(0),
            #                              os.path.join(cur_path, f'cam_{model_name}_conv1.png'))
            torchvision.utils.save_image(visualization_torch.permute(2, 0, 1).unsqueeze(0),
                                         os.path.join(cur_path, f'cam_{model_name}_layer3.png'))


if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()


