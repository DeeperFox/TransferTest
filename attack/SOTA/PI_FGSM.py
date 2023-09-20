"""
Patch-wise Attack for Fooling Deep Neural Network
paper: https://arxiv.org/abs/2007.06765
code: https://github.com/qilong-zhang/Patch-wise-iterative-attack
"""

import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base
from torch.nn import functional as F
import numpy as np


class PI_FGSM(Attack_Base):
    def __init__(self, model, eps, alpha, iters, amplification=10, kern_size=3, max_value=1., min_value=0):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = iters
        self.alpha = alpha
        self.amplification = amplification
        self.alpha_beta = alpha * amplification
        self.gamma = self.alpha_beta
        self.stack_kern, self.padding_size = self.project_kern(kern_size)

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).to(self.device)
        return stack_kern, kern_size // 2

    def project_noise(self, x, stack_kern, padding_size):
        x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
        return x

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = data.clone().detach()
        loss_func = nn.CrossEntropyLoss()
        amplification_ = torch.zeros_like(data, device=self.device)

        for _ in range(self.iters):
            adv_images.requires_grad = True
            output = self.model(adv_images)
            loss = loss_func(output, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]

            # PI-FGSM
            amplification_ += self.alpha_beta * torch.sign(grad)
            cut_noise = torch.clamp(abs(amplification_) - self.eps, min=0., max=10000.0) * torch.sign(amplification_)
            projection = self.gamma * torch.sign(self.project_noise(cut_noise, self.stack_kern, self.padding_size))
            amplification_ += projection

            adv_images = adv_images.detach() + self.alpha_beta * torch.sign(grad) + projection
            delta = torch.clamp(adv_images - data, -self.eps, self.eps)
            adv_images = torch.clamp(data + delta, 0., 1.)

        return adv_images

