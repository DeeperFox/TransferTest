import torch
import torch.nn.functional as F
from attack.Attack_Base import Attack_Base


class Patch_mask(Attack_Base):
    def __init__(self, model, model_t, eps, alpha, iters, T=1, beta=1, max_value=1., min_value=0., decay=0.):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.alpha = alpha
        self.iters = iters
        self.T = T
        self.beta = beta
        self.decay = decay

        # teacher
        self.teacher_model = model_t

    def attack(self, images, labels, idx=-1):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)

        for _ in range(self.iters):
            adv_images.requires_grad = True

            # kd
            output_t = self.teacher_model(adv_images)
            output_s = self.model(adv_images)

            loss_soft = F.kl_div(F.log_softmax(output_s / self.T, dim=1), F.softmax(output_t / self.T, dim=1),
                                 reduction='batchmean') * self.T * self.T
            loss_hard = F.cross_entropy(output_s, labels)
            loss = loss_hard - self.beta * loss_soft

            grad = torch.autograd.grad(loss, adv_images)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = self.get_adv_example(images, adv_images, grad)

        return adv_images

