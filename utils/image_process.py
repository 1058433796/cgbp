import os
import numpy as np
from torchvision import transforms
from utils import config
from PIL import Image
import torch
def min_max_normalize(attr):
    if isinstance(attr, np.ndarray):
        if attr.max() == attr.min():
            return attr
        attr -= attr.min()
        attr /= np.quantile(attr, 0.99)
        attr = np.clip(attr, 0., 1.)
    elif isinstance(attr, torch.Tensor):
        if attr.max() == attr.min():
            return attr
        attr -= attr.min()
        attr /= torch.quantile(attr, 0.99)
        attr = torch.clip(attr, 0., 1.)
    return attr

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (experiment1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    # grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def transform_image(img):
    resize_size = int(config['resize_size'])
    crop_size = int(config['crop_size'])
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),  # 调整图像大小
        transforms.CenterCrop((crop_size, crop_size)),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    img_as_tensor = transform(img)
    return img_as_tensor


def preprocess_image(filename):
    img = Image.open(filename)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_as_tensor = transform_image(img)
    return img_as_tensor


def tensor_to_image(img):
    # 假设tensor的形状为[experiment1, 3, 224, 224]
    # 首先移除批次维度，因为我们只处理单个图像
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if img.dim() == 4:
        img = img.squeeze()

    # 反标准化
    # 对于每个通道，我们需要将标准化的值转换回原始的像素值
    # 这里的mean和std应与预处理时使用的值相同
    device = img.device
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    img = img * std[:, None, None] + mean[:, None, None]

    # 裁剪和缩放的逆操作不是必需的，因为它们不会改变像素值

    # 将张量转换为PIL图像
    # 首先将张量转换为0-255的值
    img = torch.clamp(img * 255, min=0, max=255)
    # 转换为uint8类型
    img = img.byte()
    # 转换为PIL图像，注意需要转置维度，因为PIL图像期望的是（宽，高，通道）
    img = Image.fromarray(img.cpu().numpy().transpose(1, 2, 0), 'RGB')

    return img

