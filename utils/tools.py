import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchattacks
import yaml
from easydict import EasyDict
from tqdm import tqdm


def kl_div(input, target):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    return kl_loss(F.log_softmax(input), F.log_softmax(target))


def tensor2np(input):
    output = (input - input.min()) / (input.max() - input.min())
    return output.cpu().detach().numpy()


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
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


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_exp_name(attack_method, dataset, config):
    # time
    curr_time = time.strftime("%m%d%H%M")

    # attack method
    if attack_method == 'fgsm':
        exp_name = 'FGSM'
    elif attack_method == 'pgd':
        exp_name = f'PGD_i{config.ADV.iters}'
    elif attack_method == 'mi-fgsm':
        exp_name = f'MI-FGSM_i{config.ADV.iters}_m{config.ADV.MI.decay}'
    elif attack_method == 'di-fgsm':
        exp_name = f'DI-FGSM_i{config.ADV.iters}'
    elif attack_method == 'i-fgsm2':
        exp_name = f'I-FGSM_i{config.ADV.iters}'
    elif attack_method == 'ours':
        exp_name = f'OURS_i{config.ADV.iters}'
    elif attack_method == 'pi-fgsm':
        exp_name = f'PI-FGSM_i{config.ADV.iters}_a{config.ADV.PI.amplification}_k{config.ADV.PI.kern_size}'
    elif attack_method == 'ti-fgsm':
        exp_name = f'TI-FGSM_i{config.ADV.iters}_k{config.ADV.TI.len_kernel}'
    elif attack_method == 'ni-fgsm':
        exp_name = f'NI-FGSM_i{config.ADV.iters}'
    elif attack_method == 'sini-fgsm':
        exp_name = f'SINI-FGSM_i{config.ADV.iters}_m{config.ADV.SINI.m}'
    elif attack_method == 'vmi-fgsm':
        exp_name = f'VMI-FGSM_i{config.ADV.iters}_n{config.ADV.VMI.N}_b{config.ADV.VMI.beta}'
    elif attack_method == 'vni-fgsm':
        exp_name = f'VNI-FGSM_i{config.ADV.iters}_n{config.ADV.VMI.N}_b{config.ADV.VMI.beta}'
    elif attack_method == 'admix':
        exp_name = f'Admix_i{config.ADV.iters}_m1-{config.ADV.Admix.m_1}_m2-{config.ADV.Admix.m_2}' \
                   f'_eta{config.ADV.Admix.eta}'
    elif attack_method == 'smi-fgsm':
        exp_name = f'SMI-FGSM_i{config.ADV.iters}_m{config.ADV.MI.decay}'
    elif attack_method == 'di-fgsm2':
        exp_name = f'DI-FGSM_i{config.ADV.iters}'

    #################################################################
    #                            Test                               #
    #################################################################
    elif attack_method == 'lafa':
        exp_name = f'LAFA_i{config.ADV.iters}_b{config.ADV.KD.beta}_T{config.ADV.KD.T}_Start{config.ADV.KD.start_kd}'
    elif attack_method == 'lafa_ni':
        exp_name = f'LAFA-NI_i{config.ADV.iters}_b{config.ADV.KD.beta}_T{config.ADV.KD.T}_Start{config.ADV.KD.start_kd}'
    elif attack_method == 'lafa_ti':
        exp_name = f'LAFA-TI_i{config.ADV.iters}_b{config.ADV.KD.beta}_T{config.ADV.KD.T}_Start{config.ADV.KD.start_kd}' \
                   f'_k{config.ADV.TI.len_kernel}'
    elif attack_method == 'lafa_di':
        exp_name = f'LAFA-DI_i{config.ADV.iters}_b{config.ADV.KD.beta}_T{config.ADV.KD.T}_Start{config.ADV.KD.start_kd}'
    elif attack_method == 'lafa_vmi':
        exp_name = f'LAFA-VMI_i{config.ADV.iters}_b{config.ADV.KD.beta}_T{config.ADV.KD.T}_Start{config.ADV.KD.start_kd}'
    elif attack_method == 'pi2-fgsm':
        exp_name = f'PI2-FGSM_i{config.ADV.iters}'
    elif attack_method == 'DeepAttack':
        exp_name = f'DeepAttack_i{config.ADV.iters}'
    elif attack_method == 'DeepAttack_DI':
        exp_name = f'DeepAttack-DI_i{config.ADV.iters}'
    elif attack_method == 'DeepAttack_NI':
        exp_name = f'DeepAttack-NI_i{config.ADV.iters}'
    elif attack_method == 'DeepAttack_TI':
        exp_name = f'DeepAttack-TI_i{config.ADV.iters}'
    elif attack_method == 'DeepAttack_VMI':
        exp_name = f'DeepAttack-VMI_i{config.ADV.iters}'
    elif attack_method == 'exp-fgsm':
        exp_name = f'Expain-FGSM_i{config.ADV.iters}'
    elif attack_method == 'Freq':
        exp_name = f'Freq_i{config.ADV.iters}'
    else:
        raise 'no match at method'
    exp_name += f'_{dataset}_{config.model}_s{config.ADV.seed}_t{curr_time}'
    return exp_name


def get_save_file_path(cfg):
    return get_exp_name(cfg.method, cfg.dataset, cfg) + '.csv'


def save_metrix(cfg, metrix):
    header = ['model_name']
    clean_acc = ['clean_acc']
    rob_acc = ['rob_acc']
    attack_rate = ['attack_rate']

    save_path = cfg.PATH.log
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for model_name, value in metrix.items():
        header.append(model_name)
        clean_acc.append(str(value.clean_acc))
        rob_acc.append(str(value.adv_acc))
        attack_rate.append(str(value.attack_rate))

    with open(os.path.join(save_path, get_save_file_path(cfg)), 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerow(clean_acc)
        f_csv.writerow(rob_acc)
        f_csv.writerow(attack_rate)


def get_project_path():
    """得到项目路径"""
    project_path = os.path.join(
        os.path.dirname(__file__),
        "..",
    )
    return os.path.abspath(project_path)


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    if not 'model' in config.keys():
        config.model = f'{config.model_1};{config.model_2};{config.model_3};{config.model_4}'

    # Add the output path
    config.exp_name = get_exp_name(args.method, args.dataset, config)
    config.PATH.root = get_project_path()
    config.PATH.log = os.path.join(config.PATH.root, config.PATH.log)
    config.PATH.dataset = os.path.join(config.PATH.root, config.PATH.dataset)
    config.PATH.checkpoint = os.path.join(config.PATH.root, config.PATH.checkpoint)
    config.ADV.eps = config.ADV.eps / 255.
    config.ADV.alpha = config.ADV.alpha / 255.

    return config


def test(net, test_loader, device):
    print('==> Teacher validation')
    nat_correct = 0
    rob_correct = 0
    total = 0
    attack_method = torchattacks.PGD(model=net, eps=8 / 255, alpha=2 / 255, steps=20)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.enable_grad():
                adv_inputs = attack_method(inputs, targets)
            nat_outputs = net(inputs)
            rob_outputs = net(adv_inputs)

            _, nat_predicted = nat_outputs.max(1)
            _, rob_predicted = rob_outputs.max(1)
            total += targets.size(0)

            nat_correct += nat_predicted.eq(targets).sum().item()
            rob_correct += rob_predicted.eq(targets).sum().item()

    print('Nat. Acc: %.3f%% (%d/%d) | Rob. Acc: %.3f%% (%d/%d)' %
          (100. * nat_correct / total, nat_correct, total, 100. * rob_correct / total, rob_correct, total))



