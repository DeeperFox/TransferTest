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


def project_noise(x, stack_kern, padding_size):
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


def creat_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    return gauss


class PI2_FGSM(Attack_Base):
    def __init__(self, model, eps, alpha, iters, kern_size=3, max_value=1., min_value=0):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = iters
        self.alpha = alpha
        self.stack_kern, self.padding_size = self.project_kern(kern_size)
        self.eps_2 = alpha

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern, kern_size // 2

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = data.clone().detach()
        loss_func = nn.CrossEntropyLoss()

        for _ in range(self.iters):
            adv_images.requires_grad = True
            output = self.model(adv_images)
            loss = loss_func(output, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]

            # PI-FGSM
            cut_noise = torch.clamp()


            adv_images = adv_images.detach() + self.alpha_beta * torch.sign(grad) + projection
            delta = torch.clamp(adv_images - data, -self.eps, self.eps)
            adv_images = torch.clamp(data + delta, 0., 1.)

        return adv_images

