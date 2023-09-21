import torch
import torch.nn as nn
import timm  # 导入timm库，该库提供了大量预训练模型
from masking import get_masking  # 导入获取遮罩的函数
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchattacks.attack import Attack  # 导入对抗攻击库
import os
from torchvision import datasets, transforms
from torchvision.utils import save_image  # 用于保存图像的函数

class Context_change:
    print('')

    def __init__(self, model=None, method="CtxMaskSingleFill(0.0001)", prob=0.5, save_mask=False, dataset=None):
        # 如果未提供模型，则使用预训练的ResNet18模型
        if model is None:
            self.model = timm.create_model("resnet18", pretrained=True)
        else:
            self.model = model
        self.model.eval()  # 设置模型为评估模式，用于推断
        self.method = method  # 设置数据增强方法
        self.prob = prob  # 设置数据增强的概率
        self.save_mask = save_mask  # 是否保存遮罩
        self.masking = get_masking(method, model=self.model, save_mask=save_mask)  # 获取遮罩方法
        self.device = next(self.model.parameters()).device  # 获取模型所在的设备，例如GPU或CPU
        self.dataset = dataset  # 设置数据集
        self.cam_masking = get_masking("CAMMasking", model=self.model)  # 获取CAM遮罩方法

    def augment(self, random_image, random_label,images, labels=None):
        # 对输入图像进行数据增强
        images = images.to(self.device)  # 将图像移到设备上
        labels = labels.to(self.device)  # 将标签移到设备上
        orig_mask = self.cam_masking(images, labels)  # 为原始图像生成CAM遮罩
        fill_image, fill_label = random_image, random_label  # 从数据集中随机获取一个填充图像和其标签
        fill_image = fill_image.unsqueeze(0).to(self.device)  # 增加维度并移至设备上
        fill_label = torch.tensor([fill_label], dtype=torch.long).to(self.device)  # 转换填充标签并移至设备上
        fill_mask = self.cam_masking(fill_image, fill_label)  # 使用其自己的标签为填充图像生成CAM遮罩
        torch.cuda.empty_cache()
        images = images * orig_mask  # 使用CAM遮罩保留原始图像中的主要主题
        fill_image = fill_image * (~fill_mask)  # 反转填充遮罩以保留背景
        augmented_images = self.masking(images, labels, fill_image)  # 应用遮罩方法
        return augmented_images

    def save(self, path):
        # 保存模型的权重
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # 加载模型的权重
        self.model.load_state_dict(torch.load(path))

# # 加载数据集
# data_path = 'Sub_imagenet'  # 数据集路径
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
# augmenter = Context_change(model=None, prob=0.5,dataset=dataset)
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
