import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

import torchattacks
import numpy as np
import argparse


def parser():
    parser = argparse.ArgumentParser(description='AT')
    parser.add_argument('--attack_method', required=True)
    parser.add_argument('--model_name', type=str, default='wideresnet')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', default=128)
    return parser.parse_args()


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def valid(args, model, valid_loader, adv_test=False, use_pseudo_label=False):
    total_acc = 0.
    num = 0
    total_adv_acc = 0.

    if args.attack_method == 'pgd':
        attack_method = torchattacks.PGD(model=model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
    elif args.attack_method == 'aa':
        attack_method = torchattacks.AutoAttack(model, eps=8/255, n_classes=10 if args.dataset == 'cifar10' else 100)

    with torch.no_grad():
        for idx, (data, label) in enumerate(tqdm(valid_loader)):
            data, label = data.to(device), label.to(device)

            # output = t(f(data))
            output = model(data)

            pred = torch.max(output, dim=1)[1]
            std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

            total_acc += std_acc
            num += output.shape[0]

            if adv_test:
                with torch.enable_grad():
                    adv_data = attack_method(data,
                                             pred if use_pseudo_label else label)
                # adv_output = t(f(adv_data))
                adv_output = model(adv_data)

                adv_pred = torch.max(adv_output, dim=1)[1]
                adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                total_adv_acc += adv_acc
            else:
                total_adv_acc = -num

    return total_acc / num, total_adv_acc / num


if __name__ == '__main__':
    args = parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.Resize(size=32, antialias=True),
            transforms.ToTensor()
        ])
        test_set = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.Resize(size=32, antialias=True),
            transforms.ToTensor()
        ])
        test_set = datasets.CIFAR100(root='./data/', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToTensor()
        ])
        test_set = datasets.ImageFolder(root='/dataset/ImageNet/val', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=4)

    # load model
    import timm
    if args.model_name == 'resnet18':
        model = timm.create_model('resnet18', num_classes=1000,
                                  checkpoint_path='./checkpoints/pretrain_at/imagenet/resnet18.pth')
    elif args.model_name == 'deit_t':
        model = timm.create_model('deit_tiny_patch16_224', num_classes=1000,
                                  checkpoint_path='./checkpoints/pretrain_at/imagenet/deitt.pth')

    model.to(device)
    model.eval()

    # loss function
    loss_func = nn.CrossEntropyLoss()

    clean_acc, adv_acc = valid(args, model, test_loader, True)

    inf = f'RESULT:\n' \
          f'clean acc: {clean_acc}\n' \
          f'{args.attack_method} acc:   {adv_acc}\n'

    print(inf)




