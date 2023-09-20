import timm
import torch
import numpy as np
import os

import torchattacks
import torchvision.utils
from captum.attr import IntegratedGradients, Saliency, DeepLiftShap, GuidedBackprop, GuidedGradCam, \
    NoiseTunnel
from torchvision import datasets, transforms


def norm(data):
    return (data - data.min()) / (data.max() - data.min())

def main():
    device = torch.device('cuda:0')
    # Dataloader
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
    ])
    test_set = datasets.ImageFolder(root=os.path.join('/dataset', 'ImageNet', 'val'), transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    # Model
    model = timm.create_model('resnet18', pretrained=True)
    model.to(device)
    model.eval()

    attack_method = torchattacks.PGD(model, eps=16/255, alpha=1.6/255, steps=10)

    baseline = torch.zeros((8, 3, 224, 224), device=device)
    # Integrated Gradients
    ig = IntegratedGradients(model)
    # Saliency
    sl = Saliency(model)
    # DeepLiftShap
    shap = DeepLiftShap(model)
    # Guide Backprop
    guide_backprop = GuidedBackprop(model)
    # Guide Grad CAM
    guide_cam = GuidedGradCam(model, model.layer4)
    # Noise Tunnel
    nt = NoiseTunnel(shap)

    result = []

    for idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        result.append(data.cpu())
        # Integrated Gradients
        ig_map = ig.attribute(data, baseline, target=label)
        result.append(norm(ig_map.cpu()))
        # Saliency
        sl_map = sl.attribute(data, target=label)
        result.append(norm(sl_map.cpu()))
        # DeepLiftShap
        shap_map = shap.attribute(data, baseline, target=label)
        result.append(norm(shap_map.cpu()))
        # Guide Backprop
        prop_map = guide_backprop.attribute(data, target=label)
        result.append(norm(prop_map.cpu()))
        # Guide CAM
        cam_map = guide_cam.attribute(data, target=label)
        result.append(norm(cam_map.cpu()))
        # # Shapely Value Sampling
        # svs_map = svs.attribute(data, baseline, target=label)
        # result.append(norm(svs_map))

        print(idx)
        if idx == 64:
            break

    result = torch.cat(result)
    torchvision.utils.save_image(result, './toy_exp.png', nrow=8)
    print('Finish')




if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    main()


