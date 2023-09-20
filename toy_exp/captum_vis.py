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

from captum.attr import IntegratedGradients, GuidedBackprop, Saliency, GuidedGradCam, InputXGradient, DeepLift


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
    vis_model = models[configs.model]

    # incv3_at = timm.create_model('inception_v3', checkpoint_path='../checkpoints/pretrain_at/imagenet/inception_v3.pth')
    # incv3_at.to(device)
    # incv3_at.eval()
    incv3_at = timm.create_model('resnet18', checkpoint_path='../checkpoints/pretrain_at/imagenet/resnet18.pth')
    incv3_at.to(device)
    incv3_at.eval()

    ig = IntegratedGradients(vis_model)
    gbp = GuidedBackprop(vis_model)
    sali = Saliency(vis_model)
    ggc = GuidedGradCam(vis_model, vis_model[-1].Mixed_7c)
    ixg = InputXGradient(vis_model)
    dls = DeepLift(vis_model)

    path = './explain'

    for idx, (data, label) in enumerate(tqdm(test_loader)):
        data, label = data.to(device), label.to(device)

        # IntegratedGradients
        attr_ig = ig.attribute(data, baselines=torch.zeros_like(data, device=device), target=label)
        # GuidedBackProp
        attr_gbp = gbp.attribute(data, target=label)
        # Saliency
        attr_s = sali.attribute(data, target=label)
        # GuidedGradCam
        attr_ggc = ggc.attribute(data, target=label)
        # InputXGradient
        attr_ixg = ixg.attribute(data, target=label)
        # DeepLiftShift
        attr_dls = dls.attribute(data, target=label, baselines=torch.zeros_like(data, device=device))

        # AT
        data.requires_grad = True
        output_at = incv3_at(data)
        loss = torch.nn.CrossEntropyLoss()(output_at, label)
        grad_at = torch.autograd.grad(loss, data)[0]

        data.requires_grad = True
        output_at = vis_model(data)
        loss = torch.nn.CrossEntropyLoss()(output_at, label)
        grad_nt = torch.autograd.grad(loss, data)[0]

        cur_path = os.path.join(path, f'{label.item()}')
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

        torchvision.utils.save_image(data, os.path.join(cur_path, 'image.png'), normalize=True)
        torchvision.utils.save_image(attr_ig, os.path.join(cur_path, 'IntegratedGradients.png'), normalize=True)
        torchvision.utils.save_image(attr_gbp, os.path.join(cur_path, 'GuidedBackProp.png'), normalize=True)
        torchvision.utils.save_image(attr_s, os.path.join(cur_path, 'Saliency.png'), normalize=True)
        torchvision.utils.save_image(attr_ggc, os.path.join(cur_path, 'GuidedGradCam.png'), normalize=True)
        torchvision.utils.save_image(attr_ixg, os.path.join(cur_path, 'InputXGradient.png'), normalize=True)
        torchvision.utils.save_image(attr_dls, os.path.join(cur_path, 'DeepLiftShift.png'), normalize=True)
        torchvision.utils.save_image(grad_at, os.path.join(cur_path, 'grad_at.png'), normalize=True)
        torchvision.utils.save_image(grad_nt, os.path.join(cur_path, 'grad_nt.png'), normalize=True)

if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()



