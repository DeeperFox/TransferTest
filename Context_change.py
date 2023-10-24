import torch
import torch.nn as nn
import timm
from masking import get_masking
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchattacks.attack import Attack
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image
import glob
from PIL import Image

class Context_change:
    def __init__(self, model=None, method="CtxMaskSingleFill(0.9)", prob=0.5, save_mask=False, dataset=None):
        # 如果未提供模型，则使用预训练的ResNet18模型
        if model is None:
            self.model = timm.create_model("resnet18", pretrained=True)
        else:
            self.model = model
        self.model.eval()
        self.method = method
        self.prob = prob
        self.save_mask = save_mask
        self.masking = get_masking(method, model=self.model, save_mask=save_mask)
        self.device = next(self.model.parameters()).device
        self.dataset = dataset

        # 加载背景图像文件夹中的所有图像
        self.background_images = [Image.open(file).convert("RGB") for file in glob.glob('background_image/*.jpg')]
        self.background_images = [TF.to_tensor(bg_image.resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(self.device) for bg_image in self.background_images]

    def augment(self, images, labels, cam0, flag):
        images = images.to(self.device)
        labels = labels.to(self.device)  
        # orig_mask = self.cam_masking(images, labels)  # 为原始图像生成CAM遮罩
        fill_image = self.background_images[torch.randint(0, len(self.background_images), (1,))]  # 从背景图像列表中随机选择一个图像
        fill_image = fill_image.to(self.device)
        # images = images * orig_mask  # 使用CAM遮罩保留原始图像中的主要主题
        augmented_images, cam = self.masking(images, labels, cam0, flag, fill_image)  # 应用遮罩方法
        return augmented_images, cam

    def save(self, path):
        # 保存模型的权重
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # 加载模型的权重
        self.model.load_state_dict(torch.load(path))

# # 加载数据集
# data_path = 'data/Sub_imagenet'  # 数据集路径
# dataset = datasets.ImageFolder(
#     root=data_path,
#     transform=transforms.Compose([
#         transforms.Resize((224, 224)),  # 将图像大小调整为224x224
#         transforms.ToTensor(),  # 将图像转换为张量
#     ])
# )
# loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # 创建数据加载器
#
# # 初始化模型和增强器
# augmenter =  Context_change(model=None, prob=0.5,dataset=dataset)
#
# def adjust_augmented_shape(augmented_inputs):
#     # 调整增强后输入的形状
#     batch_size, num_images, channels, height, width = augmented_inputs.shape
#     adjusted_augmented_inputs = augmented_inputs.reshape(batch_size * num_images, channels, height, width)
#     return adjusted_augmented_inputs
#
# # 处理并保存增强后的图片
# output_dir = 'augmented_images'  # 输出目录
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)  # 如果输出目录不存在，则创建它
#
# for idx, (inputs, labels) in enumerate(loader):  # 遍历数据加载器
#     print(f"Processing batch {idx + 1} of {len(loader)}")  # 打印当前批次
#     augmented_inputs = augmenter.augment(inputs, labels)  # 对输入进行数据增强
#     augmented_inputs = augmented_inputs[0]
#     if isinstance(augmented_inputs, tuple):
#         augmented_inputs = augmented_inputs[0]
#     for j, aug_img in enumerate(augmented_inputs):  # 遍历增强后的图像
#         print(f"Saving image {j + 1} of {len(augmented_inputs)} in batch {idx + 1}")  # 打印当前图像编号
#         save_path = os.path.join(output_dir, f"augmented_{idx * loader.batch_size + j}.png")  # 设置保存路径
#         save_image(aug_img, save_path)  # 保存图像
#
# print("增强的图片已保存至augmented_images文件夹。")  # 打印完成消息

# python Context_change.py
