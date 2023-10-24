import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base
from torch.nn import functional as F
import numpy as np


class I_FGSM(Attack_Base):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, max_value=1., min_value=0, resize_rate=0.9, decay=0.0,
                 diversity_prob=0.5, random_start=False):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.iters = steps
        self.alpha = alpha
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start

    def attack(self, data, labels, idx=-1):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(data).detach().to(self.device)

        adv_images = data.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.iters):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - data, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        return adv_images
