"""
Patch-wise Attack for Fooling Deep Neural Network
paper: https://arxiv.org/abs/2007.06765
code: https://github.com/qilong-zhang/Patch-wise-iterative-attack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attack.Attack_Base import Attack_Base


class SMI_FGSM(Attack_Base):
    def __init__(self, model, eps, alpha, iters, decay, resize_rate, diversity_prob, max_value=1., min_value=0):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = iters
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = data.clone().detach()
        loss_func = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(data).detach().to(self.device)

        for _ in range(self.iters):
            adv_images.requires_grad = True
            output = self.model(self.input_diversity(adv_images))
            loss = loss_func(output, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            # momentum
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            adv_images = self.get_adv_example(data, adv_images, grad)

        return adv_images

