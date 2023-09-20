import os
import yaml
import csv
import torch
from utils.args import args

from utils.get_dataset import get_dataset
from model.get_models import get_models
from utils.tools import get_project_path

from tqdm import tqdm
from utils.AverageMeter import SimilarityMeter

import torch.nn.functional as F


def get_cos_meter():
    yaml_path = 'checkpoint.yaml'
    with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    metrix = {}
    for key_1, _ in yaml_data.items():
        metrix[key_1] = {}
        for key_2, _ in yaml_data.items():
            metrix[key_1][key_2] = SimilarityMeter()
    return metrix


def save_sim_meter(metrix):
    save_path = os.path.join(args.root_path, args.metrix_path)

    results = {}
    yaml_path = 'checkpoint.yaml'
    with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    for key, _ in yaml_data.items():
        results[key] = [key]
        for key_, _ in yaml_data.items():
            results[key].append(metrix[key][key_].avg_sim)

    with open(os.path.join(save_path, 'cos_sim.csv'), 'w', newline='') as f:
        f_csv = csv.writer(f)
        for key, _ in yaml_data.items():
            f_csv.writerow(results[key])

def main():
    device = torch.device('cuda:0')
    # get dataloader
    test_loader = get_dataset()
    # get models
    models, _ = get_models(device)
    metrix = get_cos_meter()

    for idx, (data, label) in enumerate(tqdm(test_loader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)

        for model_name, model in models.items():
            data = data.detach()
            data.requires_grad = True
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, label)
            grad = torch.autograd.grad(loss, data)[0]
            grad = F.normalize(grad, dim=1)

            for model_name_, model_ in models.items():
                data = data.detach()
                data.requires_grad = True
                output_ = model_(data)
                loss_ = torch.nn.CrossEntropyLoss()(output_, label)
                grad_ = torch.autograd.grad(loss_, data)[0]
                grad_ = F.normalize(grad_, dim=1)

                cos_sim = F.cosine_similarity(grad, grad_, dim=1, eps=1e-12)
                metrix[model_name][model_name_].update(cos_sim.mean().item(), 1)

        if idx == 2:
            break

    save_sim_meter(metrix)


if __name__ == '__main__':
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main()
