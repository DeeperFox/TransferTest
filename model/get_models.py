"""
get model to attack
22-8-16
"""
import os
import timm
import yaml
from collections import OrderedDict
import torch
from torchvision import transforms

from utils.AverageMeter import AccuracyMeter

yaml_path = '../config/checkpoint.yaml'
adv_yaml_path = '../config/checkpoint_adv.yaml'


def prepare_model(model, device, eval=False):
    model = model.to(device)
    if eval:
        model = model.eval()
    return model


def load_ckp(model, ckp_path, device):
    ckp = torch.load(ckp_path, map_location=device)
    if list(torch.load(ckp_path, map_location=device))[0].startswith("_orig_mod"):
        new_ckp = OrderedDict()
        for k, v in ckp.items():
            new_ckp[k[10:]] = v
        ckp = new_ckp
    model.load_state_dict(ckp)
    return model


def get_models(cfg, device):
    print('🌟\tBuilding models...')
    metrix, models = {}, {}

    with open(os.path.join(cfg.PATH.root, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    for key, value in yaml_data.items():
        model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
        model.eval()
        if 'inc' in key or 'vit' in key or 'bit' in key:
            models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
        else:
            models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              model)
        models[key].eval()
        metrix[key] = AccuracyMeter()
        print(f'⭐\tLoad {key} successfully')

    # load ens at model
    checkpoint_path = os.path.join(cfg.PATH.root, 'checkpoints', 'pretrain_at', 'imagenet')
    models['incv3_ens3'] = timm.create_model('inception_v3', checkpoint_path=os.path.join(checkpoint_path,
                                                                                          'inc_v3_ens3.pth')
                                             ).to(device)
    models['incv3_ens3'].eval()
    metrix['incv3_ens3'] = AccuracyMeter()
    print('⭐\tLoad incv3_ens3 successfully')

    models['incv3_ens4'] = timm.create_model('inception_v3', checkpoint_path=os.path.join(checkpoint_path,
                                                                                          'inc_v3_ens4.pth')
                                             ).to(device)
    models['incv3_ens4'].eval()
    metrix['incv3_ens4'] = AccuracyMeter()
    print('⭐\tLoad incv3_ens4 successfully')

    models['incv2_ens'] = timm.create_model('inception_resnet_v2', checkpoint_path=os.path.join(checkpoint_path,
                                                                                                'inc_v2_ens.pth')
                                            ).to(device)
    models['incv2_ens'].eval()
    metrix['incv2_ens'] = AccuracyMeter()
    print('⭐\tLoad incv2_ens successfully')

    # models['resnet18_at'] = timm.create_model('resnet18',
    #                                           checkpoint_path=os.path.join(checkpoint_path, 'resnet18.pth'))\
    #     .to(device)
    # models['resnet18_at'].eval()
    # metrix['resnet18_at'] = AccuracyMeter()
    # print('⭐\tLoad resnet18_at successfully')

    return models, metrix


def get_model_feature_only(cfg, device):
    print('🌟\tBuilding models...')
    models = {}

    with open(os.path.join(cfg.PATH.root, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    for key, value in yaml_data.items():
        if 'vit' in key or 'deit' in key or 'swin' in key:
            continue
        model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000, features_only=True).to(device)
        model.eval()
        if 'inc' in key or 'vit' in key or 'bit' in key:
            models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
        else:
            models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              model)
        print(f'⭐\tLoad {key} successfully')
    return models


def get_one_feat_extractor(cfg, device):
    with open(os.path.join(cfg.PATH.root, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    model = timm.create_model(yaml_data[cfg.model]['full_name'],
                              pretrained=True,
                              num_classes=1000,
                              features_only=True).to(device)
    model.eval()
    return model


def get_teacher_model(cfg, device):
    num_classes = 1000
    checkpoint_path = os.path.join(cfg.PATH.checkpoint, 'pretrain_at', 'imagenet')
    if cfg.ADV.KD.teacher_name == 'resnet18':
        model_t = timm.create_model('resnet18', num_classes=num_classes,
                                    checkpoint_path=os.path.join(checkpoint_path, 'resnet18.pth'))
        # model_t = timm.create_model('resnet18', pretrained=True)
        model_t = prepare_model(model_t, eval=True, device=device)
    elif cfg.ADV.KD.teacher_name == 'resnet26':
        model_t = timm.create_model('resnet26', num_classes=num_classes,
                                    checkpoint_path=os.path.join(checkpoint_path, 'resnet26.pth'))
        model_t = prepare_model(model_t, eval=True, device=device)
    elif cfg.ADV.KD.teacher_name == 'resnet34':
        model_t = timm.create_model('resnet34', num_classes=num_classes,
                                    checkpoint_path=os.path.join(checkpoint_path, 'resnet34.pth'))
        model_t = prepare_model(model_t, eval=True, device=device)
    elif cfg.ADV.KD.teacher_name == 'resnet50':
        model_t = timm.create_model('resnet50', num_classes=num_classes,
                                    checkpoint_path=os.path.join(checkpoint_path, 'resnet50.pth.tar'))
        model_t = prepare_model(model_t, eval=True, device=device)
    elif cfg.ADV.KD.teacher_name == 'incv3':
        model_t = timm.create_model('inception_v3', num_classes=num_classes,
                                    checkpoint_path=os.path.join(checkpoint_path, 'inception_v3.pth'))
        model_t = prepare_model(model_t, eval=True, device=device)
    else:
        raise NotImplemented
    return model_t
