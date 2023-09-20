import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from attack.Attack_Base import Attack_Base  # 从attack模块导入Attack_Base基类
from torch.nn import functional as F  # 导入PyTorch函数库
import numpy as np  # 导入NumPy库
from Context_change import Context_change  # 从Context_change模块导入Context_change类

class CI_FGSM(Attack_Base):  # 定义DI_FGSM类并继承Attack_Base
    def __init__(self, random_image, random_label,model, eps=8/255, alpha=2/255, steps=10, max_value=1., min_value=0, resize_rate=0.9, decay=0.0,
                 diversity_prob=0.5, random_start=False):  # 初始化方法
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)  # 调用父类的初始化方法
        self.iters = steps  # 设置迭代次数
        self.alpha = alpha  # 设置步长
        self.decay = decay  # 设置动量衰减系数
        self.resize_rate = resize_rate  # 设置图像缩放比例
        self.diversity_prob = diversity_prob  # 设置输入多样性的概率
        self.random_start = random_start  # 设置是否随机开始
        self.context_change = Context_change(model=model, prob=diversity_prob)  # 初始化Context_change类
        self.random_image = random_image
        self.random_label = random_label

    def attack(self,data, labels, idx=-1):  # 定义攻击方法
        data = data.clone().detach().to(self.device)  # 克隆数据并移动到设备
        labels = labels.clone().detach().to(self.device)  # 克隆标签并移动到设备

        loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        momentum = torch.zeros_like(data).detach().to(self.device)  # 初始化动量

        adv_images = data.clone().detach()  # 克隆原始数据

        if self.random_start:  # 如果选择随机开始
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)  # 在原始数据基础上添加随机扰动
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()  # 对扰动数据进行截断

        for _ in range(self.iters):  # 迭代指定次数
            adv_images.requires_grad = True  # 设置需要梯度
            outputs = self.model(self.context_change.augment(self.random_image,self.random_label,adv_images, labels))  # 通过模型获取输出

            cost = loss(outputs, labels)  # 计算损失

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]  # 计算梯度

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)  # 对梯度进行归一化
            grad = grad + momentum * self.decay  # 更新梯度加上动量衰减
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()  # 更新对抗样本
            delta = torch.clamp(adv_images - data, min=-self.eps, max=self.eps)  # 计算扰动范围
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()  # 对扰动数据进行截断

        return adv_images  # 返回对抗样本
