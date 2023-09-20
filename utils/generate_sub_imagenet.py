import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'classifiers')))
import torch
import argparse

import torchvision.utils

from get_dataset import get_dataset
from model.get_models import get_models
from utils.tools import parse_config_file, same_seeds, save_metrix


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


def search_imagenet(models, dataloader, device, incorrect_num, num_images, n, results=None):
    if results is None:
        results = []

    for idx, (data, label) in enumerate(dataloader):
        is_clean = True
        incorrect = 0
        data, label = data.to(device), label.to(device)
        if results[label.item()] is not None:
            continue
        for model_name, model in models.items():
            logit = model(data)
            predict = logit.max(1)[1]
            if predict != label:
                incorrect += 1
                if incorrect > incorrect_num:
                    is_clean = False
                    break
        if is_clean:
            results[label.item()] = {'data': data.cpu(), 'idx': label.item()}
            n += 1
            print(f'[Clean] {n}/{num_images} | [All] {idx}/{len(dataloader)}')
        else:
            print(f'[ Adv ] {n}/{num_images} | [All] {idx}/{len(dataloader)}')

        if n == num_images:
            break

    return results, n

def main():
    configs.DATA.batch_size = 1
    device = torch.device(f'cuda:{configs.gpu_id}')
    # get dataloader
    test_loader = get_dataset(configs)
    class_to_idx = test_loader.dataset.class_to_idx
    idx_to_class = [None] * 1000
    for k, v in class_to_idx.items():
        idx_to_class[v] = k

    # get models
    models, metrix = get_models(configs, device=device)

    num_images = 1000
    results = [None] * 1000
    n = 0

    max_incorrect_num = 10

    for i in range(max_incorrect_num):
        results, n = search_imagenet(models, test_loader, device, incorrect_num=i, num_images=num_images, n=n,
                                     results=results)
        print(f'Find {n} images -- incorrect: {i}')

        if n == num_images:
            break

    path = './Sub_imagenet'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(1000):
        cur_path = os.path.join(path, idx_to_class[results[i]['idx']])
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        torchvision.utils.save_image(results[i]['data'], os.path.join(cur_path, f'{i}.png'))

    print('finish')



if __name__ == '__main__':
    same_seeds(configs.ADV.seed)
    main()