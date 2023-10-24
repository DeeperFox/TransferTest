import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base
from torch.nn import functional as F
import numpy as np
from Context_change import Context_change
import os
from torchvision.utils import save_image
from datetime import datetime

class CI_FGSM(Attack_Base):  # 定义DI_FGSM类并继承Attack_Base
    def __init__(self, dataset,model, eps=8/255, alpha=2/255, steps=10, max_value=1., min_value=0, resize_rate=0.9, decay=0.0,
                 diversity_prob=0.5, random_start=False):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = steps
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.dataset = dataset
        self.context_change = Context_change(model=model, prob=diversity_prob,dataset=self.dataset)


    def attack(self,data, labels, idx=-1):  # 定义攻击方法
        data = data.clone().detach().to(self.device)  # 克隆数据并移动到设备
        labels = labels.clone().detach().to(self.device)  # 克隆标签并移动到设备

        loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        momentum = torch.zeros_like(data).detach().to(self.device)  # 初始化动量

        adv_images = data.clone().detach()  # 克隆原始数据

        if self.random_start:  # 如果选择随机开始
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)  # 在原始数据基础上添加随机扰动
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # 对扰动数据进行截断

        flag = True
        cam0 = None

        for _ in range(self.iters):  # 迭代指定次数
            adv_images.requires_grad = True
            # print(3,adv_images.shape)
            aug_img, cam0 = self.context_change.augment(adv_images, labels, cam0, flag)
            flag = False
            if not os.path.exists('mask_img'):
                os.makedirs('mask_img')
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            save_path = os.path.join('mask_img', 'masked_image_{}.png'.format(timestamp))
            save_image(aug_img[0], save_path)

            # adv_images.detach_()
            aug_img = aug_img.detach()
            aug_img.requires_grad = True
            aug_img = aug_img.squeeze(1)
            outputs = self.model(aug_img)  # 通过模型获取输出
              # 设置需要梯度
            # print(4, outputs.shape)

            cost = loss(outputs, labels)  # 计算损失

            #print(cost.grad_fn)
            grad = torch.autograd.grad(cost, aug_img, retain_graph=False, create_graph=False)[0]  # 计算梯度

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)  # 对梯度进行归一化
            grad = grad + momentum * self.decay  # 更新梯度加上动量衰减
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()  # 更新对抗样本
            delta = torch.clamp(adv_images - data, min=-self.eps, max=self.eps)  # 计算扰动范围
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()  # 对扰动数据进行截断

        if not os.path.exists('adv_images'):
            os.makedirs('adv_images')

            # 使用当前时间戳作为文件名的唯一后缀
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        save_path = os.path.join('adv_images', 'adv_image_{}.png'.format(timestamp))
        save_image(adv_images[0], save_path)

        return adv_images  # 返回对抗样本
