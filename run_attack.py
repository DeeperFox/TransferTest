import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# 将当前脚本所在的目录添加到系统路径。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'classifiers')))
# 将当前脚本所在目录下的'model/classifiers'添加到系统路径。

import argparse
# 导入argparse模块，用于解析命令行参数。
import torch
# 导入PyTorch深度学习库。
from tqdm import tqdm
# 导入tqdm，用于显示进度条。
from utils.get_attack import get_attack
# 从'utils'中导入'get_attack'函数。
from utils.get_dataset import get_dataset
# 从'utils'中导入'get_dataset'函数。
from model.get_models import get_models, get_teacher_model
# 从'model'中导入'get_models'和'get_teacher_model'函数。
from utils.tools import parse_config_file, same_seeds, save_metrix
# 从'utils/tools'中导入工具函数。
import warnings
# 导入warnings模块。
warnings.filterwarnings("ignore")
# 抑制所有警告。

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # 创建命令行参数解析器，并提供描述信息。
    parser.add_argument('-c', '--config', default='config/configs.yml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yml)')
    # 添加命令行参数，指定配置文件的路径，默认为'configs.yml'。
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='training dataset, e.g. cifar10/100 or imagenet')
    # 添加命令行参数，指定训练数据集，默认为'imagenet'。
    parser.add_argument('--model', required=True, help='the model for black-box attack')
    # 添加必需的命令行参数，指定用于黑盒攻击的模型。
    parser.add_argument('-m', '--method', default='standard', type=str,
                        help='at method, e.g. standard at, trades, mart, etc.')
    # 添加命令行参数，指定攻击方法，默认为'standard'。
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu id used for training.')
    # 添加命令行参数，指定用于训练的GPU ID，默认为0。
    return parser.parse_args()
    # 返回解析后的命令行参数。

# 解析配置文件并初始化日志记录。
configs = parse_config_file(parse_args())



def main():
    print("hello")

    device = torch.device(f'cuda:{configs.gpu_id}')
    # 根据配置的GPU ID设置设备。
    test_loader = get_dataset(configs)
    # 获取数据集。
    models, metrix = get_models(configs, device=device)
    # 获取模型和评价指标。

    print(f'surrogate model: {configs.model}\tattack method: {configs.method}')
    # 打印使用的替代模型和攻击方法。

    for idx, (data, label) in enumerate(tqdm(test_loader)):
        # 遍历数据加载器中的数据。
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        # 将数据和标签移动到指定设备。
        attack_method = get_attack(configs, model=models[configs.model], device=device)
        # 获取攻击方法。
        adv_exp = attack_method(data, label)
        # 生成对抗样本。

        for model_name, model in models.items():
            # 遍历所有模型。
            with torch.no_grad():
                # 禁用梯度计算。
                r_clean = model(data)
                r_adv = model(adv_exp)
                # 获取模型对原始数据和对抗样本的预测结果。
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # 计算原始数据的预测正确数。
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            # 计算对抗样本的预测正确数。

            metrix[model_name].update(correct_clean, correct_adv, n)
            # 更新评价指标。

    save_metrix(configs, metrix)
    # 保存评价指标。

if __name__ == '__main__':
    print(torch.cuda.is_available())

    # 如果此脚本作为主程序运行，则执行以下代码。
    same_seeds(configs.ADV.seed)
    # 设置随机种子，确保实验的可重复性。
    main()
    # 执行主函数。

# python run_attack.py --model resnet18 --method ours --dataset sub_imagenet --config ./config/hyper_params_imagenet.yml
# conda activate pytorch
# cd work2/Jiangshan/at/TransferAttack

