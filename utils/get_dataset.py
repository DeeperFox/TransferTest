import os
import torch
from torchvision import datasets, transforms


def get_dataset(cfg):
    cfg.DATA.num_classes = 1000
    transform_test = transforms.Compose([
        transforms.Resize((cfg.DATA.image_size, cfg.DATA.image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATA.mean, std=cfg.DATA.std)
    ])
    if cfg.dataset == 'imagenet':
        test_set = datasets.ImageFolder(root=os.path.join(cfg.PATH.dataset, 'ImageNet', 'val'),
                                        transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.DATA.batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=cfg.DATA.num_workers)
    elif cfg.dataset == 'sub_imagenet':
        test_set = datasets.ImageFolder(root=os.path.join(cfg.PATH.dataset, 'Sub_imagenet'),
                                        transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.DATA.batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=cfg.DATA.num_workers)
    else:
        raise NotImplemented

    return test_loader
