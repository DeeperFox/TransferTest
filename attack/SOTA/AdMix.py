"""
Patch-wise Attack for Fooling Deep Neural Network
paper: https://arxiv.org/abs/2007.06765
code: https://github.com/qilong-zhang/Patch-wise-iterative-attack
"""

import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base


class Admix(Attack_Base):
    def __init__(self, model, eps, alpha, iters, max_value=1., min_value=0, decay=0.0, m_1=5, m_2=3, eta=0.2):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = iters
        self.alpha = alpha
        self.decay = decay

        self.m_1 = m_1
        self.m_2 = m_2
        self.eta = eta

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = data.clone().detach()
        loss_func = nn.CrossEntropyLoss(reduction='sum')

        momentum = torch.zeros_like(data).detach().to(self.device)

        for _ in range(self.iters):
            adv_images.requires_grad = True
            output = self.model(adv_images)
            input_grad = 0
            for c in range(self.m_1):
                data_other = data[torch.randperm(data.shape[0])].view(data.size())
                logits = self.model(adv_images + self.eta * data_other)
                loss = loss_func(logits, labels)
                loss.backward()
                input_grad += adv_images.grad.clone()
            input_grad = input_grad / torch.mean(torch.abs(input_grad), dim=(1, 2, 3), keepdim=True)
            input_grad = input_grad + momentum * self.decay
            momentum = input_grad

            adv_images.grad.zero_()
            adv_images = self.get_adv_example(data, adv_images, input_grad)

        return adv_images

