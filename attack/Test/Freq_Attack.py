import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base
from torch.nn import functional as F
import numpy as np
import torch_dct as dct


class Freq_Attack(Attack_Base):
    def __init__(self, model, eps, alpha, iters, max_value=1., min_value=0):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = iters
        self.alpha = alpha

    def add_freq_pert(self, image, pert):
        return dct.idct_2d(self.innorm(self.norm(dct.dct_2d(image)) + pert))

    def norm(self, data):
        # self.data_max_value = data.max()
        # self.data_min_value = data.min()
        self.data_max_value = (data.reshape(10, -1).max(1)[0].view(10, 1, 1, 1) *
                               torch.ones((10, 3, 224, 224), device=self.device))
        self.data_min_value = (data.reshape(10, -1).min(1)[0].view(10, 1, 1, 1) *
                               torch.ones((10, 3, 224, 224), device=self.device))
        return (data - self.data_min_value) / (self.data_max_value - self.data_min_value)

    def innorm(self, data):
        return data * (self.data_max_value - self.data_min_value) + self.data_min_value

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = data.clone().detach()
        loss_func = nn.CrossEntropyLoss()
        pert_init_f = torch.randn(adv_images.size(), device=self.device) * 0.0001
        adv_images = self.add_freq_pert(adv_images, pert_init_f)
        adv_images = torch.clamp(adv_images, 0., 1.)
        for _ in range(self.iters):
            adv_images_tmp = adv_images.clone().detach()
            adv_images_f = dct.dct_2d(adv_images)
            adv_images_f.requires_grad = True
            adv_images = dct.idct_2d(adv_images_f)
            outputs = self.model(adv_images)
            loss = loss_func(outputs, labels)
            grads = torch.autograd.grad(loss, adv_images_f)[0]
            adv_images_f = self.innorm(self.norm(adv_images_f.detach()) + grads.sign())
            adv_images = dct.idct_2d(adv_images_f)
            # Clamp
            pert = torch.clamp(adv_images - adv_images_tmp, -self.alpha, self.alpha)
            delta = torch.clamp(adv_images_tmp + pert - data, -self.eps, self.eps)
            adv_images = torch.clamp(data + delta, 0., 1.)
        return adv_images

