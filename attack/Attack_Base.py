"""
Base implement of EnsembleAttack
"""

import torch
from abc import ABC, abstractmethod


class Attack_Base:
    def __init__(self, model, eps=8/255, max_value=1., min_value=0):
        self.model = model
        self.model.eval()

        # attack parameter
        self.eps = eps
        self.max_value = max_value
        self.min_value = min_value
        self.alpha = 0

        self.device = iter(self.model.parameters()).__next__().device

    def get_adv_example(self, ori_data, adv_data, grad):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        adv_example = adv_data.detach() + grad.sign() * self.alpha
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    @ abstractmethod
    def attack(self,
               data: torch.tensor,
               label: torch.tensor,
               idx: int = -1) -> torch.tensor:
        ...

    def __call__(self, data, label, idx=-1):
        return self.attack(data, label, idx)

