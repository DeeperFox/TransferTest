from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchattacks.attack import Attack
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re



def apply_heatmap(cam):
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 将CAM值归一化到0-1之间
    heatmap = cm.jet(cam)[..., :3]  # 使用colormap将归一化的CAM值转化为热力图
    return torch.from_numpy(heatmap.transpose(2, 0, 1))  # 转换维度并返回热力图

def get_GradCAM(model, images, labels, upsample=None):
    # Ref: https://github.com/jacobgil/pytorch-grad-cam
    images.requires_grad = True  # 确保输入图像需要梯度

    bare_model = model.module if hasattr(model, 'module') else model  # 如果模型是DataParallel，则获取其内部的原始模型
    target_layer = model[-2]  # 获取模型中的目标层

    activations = []  # 用于存储目标层的激活值的空列表

    def save_activation(module, input, output):  # 钩子函数，用于在forward时保存目标层的激活值
        activations.append(output)

    handle = target_layer.register_forward_hook(save_activation)  # 注册钩子函数到目标层
    if len(images.shape) == 5:
        images = images.squeeze(1)
    # images=images[0]
    print(6,images.shape)
    logits = model(images)
    for param in model.parameters():
        param.requires_grad = True
    handle.remove()  # 移除钩子

    loss = logits.gather(1, labels.unsqueeze(1)).sum()  # 计算损失
    grad = torch.autograd.grad(loss, activations[0],retain_graph=True)[0] # 计算关于激活值的梯度
    act = activations[0].clone().detach()  # 获取激活值
    # print(grad.shape)
    weights = grad.mean(dim=(2,3), keepdim=True)  # 计算每个通道的梯度的平均值
    cam = F.relu((weights * act).sum(dim=1))  # 计算Grad-CAM

    cam -= cam.min()  # 将CAM值归一化到0-1
    cam /= cam.max()

    cam = cam.unsqueeze(1)  # 为CAM添加一个通道维度

    if upsample:  # 如果提供了upsample参数，将CAM调整到与输入图像相同的大小
        cam = TF.resize(cam, images.size()[2:], upsample)

    output_dir = "CAMimages"  # 输出目录
    # for j, single_cam in enumerate(cam):
    #     # print(f"Saving CAM image {j + 1} of {len(cam)}")  # 输出正在保存的CAM图像编号
    #     save_path = os.path.join(output_dir, f"cam_{j}.png")
    #     single_cam = single_cam.squeeze(0)  # 去掉多余的维度
    #     cam_with_heatmap = apply_heatmap(single_cam)  # 获取热力图
    #     save_image(cam_with_heatmap, save_path)  # 保存热力图
    return cam


class CAMMaskSingleFill(Attack):  # 定义 CAMMaskSingleFill 类，继承 Attack 类

    def __init__(self, model, cam_method, threshold, ctx_mask=True, save_mask=False):
        super().__init__("CAMMaskSingleFill", model)  # 初始化父类
        self.method = cam_method  # 设置 CAM 方法（如 GradCAM 或 MMCAM）
        if cam_method == 'GradCAM':  # 如果方法是 GradCAM
            self.get_CAM = partial(
                get_GradCAM, upsample=InterpolationMode.NEAREST)  # 设置用于获取 GradCAM 的函数

        else:
            raise NotImplementedError()  # 如果方法不是上述两种，抛出异常
        self.threshold = threshold  # 设置阈值
        self.ctx_mask = ctx_mask  # 设置上下文遮罩标志
        self.save_mask = save_mask  # 设置保存遮罩标志
        self.uses_fill = True

    def forward(self, images, labels, fill_image=None):  # 定义前向传播方法
        images = images.clone().detach().to(self.device)  # 克隆并转移图像到设备
        labels = labels.clone().detach().to(self.device)  # 克隆并转移标签到设备
        print(1,images.shape)
        images = images.squeeze(0) # 移除第一个维度
        print(2,images.shape)
        cam = self.get_CAM(self.model, images, labels)  # 获取 CAM
        if self.save_mask:  # 如果需要保存遮罩
            cam, patch_cam = cam  # 分解返回的 CAM 和 patch_cam
            patch_cam = patch_cam / (patch_cam.amax(dim=1, keepdim=True) + 1e-8)  # 标准化 patch_cam
            patch_mask = (patch_cam > self.threshold)  # 创建 patch 遮罩
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)  # 标准化 CAM
        mask = (cam > self.threshold).unsqueeze(1)  # 创建 CAM 遮罩
        if self.ctx_mask:  # 如果使用上下文遮罩
            mask = ~mask  # 反转遮罩
            if self.save_mask:  # 如果需要保存遮罩
                patch_mask = ~patch_mask  # 反转 patch 遮罩
        idx = torch.randperm(images.size()[0], device=self.device)  # 随机排列索引
        if fill_image is not None:
            images_masked = images * (~mask) + fill_image * (mask)
        else:
            idx = torch.randperm(images.size()[0], device=self.device)  # 保留原来的代码，以防没有提供填充图片
            images_masked = images * (~mask) + images[idx] * (mask)
        if self.save_mask:  # 如果需要保存遮罩
            return images_masked, patch_mask  # 返回遮罩后的图像和 patch 遮罩
        return images_masked  # 返回遮罩后的图像


def get_masking(name: str = None, **kwargs):
    print(f"Received method name: {name}")

    cam_method = 'GradCAM'

    def get_params(s):
        # return the string inside the brackets
        match = re.search(r'\((.*?)\)', s)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"No matching group found in string: {s}")

    if name is None or name == 'none':
        print("Matched 'none'")
        return None

    if name.startswith('CAMMasking'):
        print("Matched 'CAMMasking'")
        return CAMMasking(kwargs['model'])


    elif name.startswith('ObjMaskSingleFill'):  # ObjMaskSingleFill(threshold)
        print("Matched '5'")
        threshold = get_params(name)
        return CAMMaskSingleFill(kwargs['model'], cam_method, float(threshold), save_mask=kwargs['save_mask'])
    elif name.startswith('CtxMaskSingleFill'):  # CtxMaskSingleFill(threshold)
        print("Matched '6'")
        threshold = get_params(name)
        return CAMMaskSingleFill(kwargs['model'], cam_method, float(threshold), ctx_mask=True)

    else:
        raise NotImplementedError()


class CAMMasking(Attack):
    def __init__(self, model, cam_method='GradCAM', threshold=0.0001):
        super(CAMMasking, self).__init__("CAMMasking", model)

        if cam_method == 'GradCAM':
            self.get_CAM = get_GradCAM

        else:
            raise NotImplementedError(f"{cam_method} is not implemented.")

        self.threshold = threshold

    def forward(self, images, labels):
        cam = self.get_CAM(self.model, images, labels)

        # Upsample CAM to match the size of images
        cam = TF.resize(cam, images.size()[2:], interpolation=InterpolationMode.BILINEAR, antialias=True)

        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        mask = (cam > self.threshold).unsqueeze(1)
        return mask