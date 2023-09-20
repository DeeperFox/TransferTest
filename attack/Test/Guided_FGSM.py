import torch
import torch.nn.functional as F
from attack.Attack_Base import Attack_Base
# from captum.attr import IntegratedGradients, GuidedBackprop, GuidedGradCam
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Guided_FGSM(Attack_Base):
    def __init__(self, model, eps, alpha, iters, max_value=1., min_value=0., decay=0.):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        self.alpha = alpha
        self.iters = iters
        self.decay = decay

        # ig
        # self.ig = IntegratedGradients(self.model)
        # self.gp = GuidedBackprop(self.model)
        # self.ggc = GuidedGradCam(self.model, self.model[-1].Conv2d_2a_3x3)

        # Guided backprop
        self.target = [self.model[-1].Conv2d_1a_3x3, self.model[-1].Conv2d_2a_3x3, self.model[-1].Conv2d_2b_3x3]
        self.cam = GuidedBackpropReLUModel(model=model, use_cuda=True)

    def attack(self, images, labels, idx=-1):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)

        for _ in range(self.iters):
            adv_images.requires_grad = True

            # loss
            output = self.model(adv_images)
            loss = F.cross_entropy(output, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            # guide backprop
            guided_grad = []
            for i in range(images.size(0)):
                guided_prop = self.cam(
                    input_img=images[i:i+1], target_category=labels[i:i+1].item())
                guided_grad.append(torch.from_numpy(guided_prop).permute(2, 0, 1).to(self.device))
            guided_grad = torch.stack(guided_grad)

            # grad
            final_grad = grad.sign() * 0 + guided_grad.sign()

            # mom
            final_grad = final_grad / torch.mean(torch.abs(final_grad), dim=(1, 2, 3), keepdim=True)
            final_grad = final_grad + momentum * self.decay
            momentum = final_grad

            adv_images = self.get_adv_example(images, adv_images, final_grad)

        return adv_images

