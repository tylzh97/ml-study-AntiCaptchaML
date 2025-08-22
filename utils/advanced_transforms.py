import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFilter
import random


class PerspectiveTransform:
    """透视变换"""
    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 定义透视变换的四个角点偏移
        offset = int(min(h, w) * self.distortion_scale)
        
        # 原始四个角点
        src_points = np.float32([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
        ])
        
        # 随机偏移后的四个角点
        dst_points = np.float32([
            [random.randint(-offset, offset), random.randint(-offset, offset)],
            [w-1 + random.randint(-offset, offset), random.randint(-offset, offset)],
            [w-1 + random.randint(-offset, offset), h-1 + random.randint(-offset, offset)],
            [random.randint(-offset, offset), h-1 + random.randint(-offset, offset)]
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        transformed = cv2.warpPerspective(img_array, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(transformed)


class ElasticTransform:
    """弹性变形"""
    def __init__(self, alpha=20, sigma=5, p=0.5):
        self.alpha = alpha  # 变形强度
        self.sigma = sigma  # 平滑参数
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            h, w, c = img_array.shape
        else:
            h, w = img_array.shape
            c = 1
        
        # 生成随机位移场
        dx = np.random.uniform(-1, 1, (h, w)) * self.alpha
        dy = np.random.uniform(-1, 1, (h, w)) * self.alpha
        
        # 高斯平滑
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
        
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = (x + dx).astype(np.float32)
        y = (y + dy).astype(np.float32)
        
        # 应用弹性变形
        transformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(transformed)


class GridDistortion:
    """网格扭曲"""
    def __init__(self, num_steps=5, distort_limit=0.3, p=0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 创建网格点
        x_step = w // self.num_steps
        y_step = h // self.num_steps
        
        # 生成控制点
        xx, yy = np.meshgrid(np.arange(0, w, x_step), np.arange(0, h, y_step))
        
        # 添加随机扭曲
        distort_x = np.random.uniform(-self.distort_limit, self.distort_limit, xx.shape) * x_step
        distort_y = np.random.uniform(-self.distort_limit, self.distort_limit, yy.shape) * y_step
        
        xx = (xx + distort_x).astype(np.float32)
        yy = (yy + distort_y).astype(np.float32)
        
        # 插值到整个图像
        map_x = cv2.resize(xx, (w, h)).astype(np.float32)
        map_y = cv2.resize(yy, (w, h)).astype(np.float32)
        
        # 应用变形
        transformed = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(transformed)


class RandomNoise:
    """添加随机噪声"""
    def __init__(self, noise_type='gaussian', intensity=0.1, p=0.5):
        self.noise_type = noise_type
        self.intensity = intensity
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.intensity, img_array.shape)
        elif self.noise_type == 'salt_pepper':
            noise = np.random.random(img_array.shape)
            noise = np.where(noise < self.intensity/2, -1, noise)
            noise = np.where(noise > 1-self.intensity/2, 1, 0)
        else:
            noise = np.random.uniform(-self.intensity, self.intensity, img_array.shape)
        
        img_array = np.clip(img_array + noise, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)


class RandomBlur:
    """随机模糊"""
    def __init__(self, blur_types=['gaussian', 'motion'], max_kernel_size=5, p=0.3):
        self.blur_types = blur_types
        self.max_kernel_size = max_kernel_size
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        blur_type = random.choice(self.blur_types)
        kernel_size = random.randint(3, self.max_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if blur_type == 'gaussian':
            return img.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
        elif blur_type == 'motion':
            # 模拟运动模糊
            img_array = np.array(img)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # 随机角度旋转内核
            angle = random.uniform(-45, 45)
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
        
        return img


class RandomShadow:
    """随机阴影"""
    def __init__(self, shadow_intensity=0.3, p=0.3):
        self.shadow_intensity = shadow_intensity
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 创建随机阴影遮罩
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 随机阴影形状和位置
        shadow_type = random.choice(['linear', 'circular', 'random'])
        
        if shadow_type == 'linear':
            # 线性渐变阴影
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            if direction == 'horizontal':
                for i in range(h):
                    mask[i, :] = i / h
            elif direction == 'vertical':
                for j in range(w):
                    mask[:, j] = j / w
            else:
                for i in range(h):
                    for j in range(w):
                        mask[i, j] = (i + j) / (h + w)
        
        elif shadow_type == 'circular':
            # 圆形阴影
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            max_radius = min(h, w) // 2
            
            for i in range(h):
                for j in range(w):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    mask[i, j] = min(1, distance / max_radius)
        
        # 应用阴影
        shadow = 1 - mask * self.shadow_intensity
        if len(img_array.shape) == 3:
            shadow = np.expand_dims(shadow, axis=2)
        
        shadowed = img_array * shadow
        shadowed = np.clip(shadowed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(shadowed)


def create_advanced_transforms(img_height=60, img_width=160, is_train=True, augment_strength='medium'):
    """
    创建高级数据增强变换
    
    Args:
        img_height: 目标图像高度
        img_width: 目标图像宽度
        is_train: 是否为训练模式
        augment_strength: 增强强度 ('light', 'medium', 'heavy')
    """
    
    if not is_train:
        # 验证/测试时只做基础变换
        return transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 根据增强强度设置参数
    strength_params = {
        'light': {
            'perspective_p': 0.2,
            'elastic_p': 0.2,
            'grid_p': 0.2,
            'noise_p': 0.3,
            'blur_p': 0.2,
            'shadow_p': 0.2,
            'color_jitter': 0.1
        },
        'medium': {
            'perspective_p': 0.4,
            'elastic_p': 0.3,
            'grid_p': 0.3,
            'noise_p': 0.4,
            'blur_p': 0.3,
            'shadow_p': 0.3,
            'color_jitter': 0.2
        },
        'heavy': {
            'perspective_p': 0.6,
            'elastic_p': 0.5,
            'grid_p': 0.5,
            'noise_p': 0.5,
            'blur_p': 0.4,
            'shadow_p': 0.4,
            'color_jitter': 0.3
        }
    }
    
    params = strength_params[augment_strength]
    
    transform_list = [
        transforms.Resize((img_height, img_width)),
        
        # 几何变换
        PerspectiveTransform(distortion_scale=0.15, p=params['perspective_p']),
        ElasticTransform(alpha=15, sigma=4, p=params['elastic_p']),
        GridDistortion(num_steps=4, distort_limit=0.2, p=params['grid_p']),
        
        # 颜色增强
        transforms.ColorJitter(
            brightness=params['color_jitter'],
            contrast=params['color_jitter'],
            saturation=params['color_jitter'],
            hue=params['color_jitter']
        ),
        
        # 噪声和模糊
        RandomNoise(noise_type='gaussian', intensity=0.08, p=params['noise_p']),
        RandomBlur(max_kernel_size=5, p=params['blur_p']),
        RandomShadow(shadow_intensity=0.2, p=params['shadow_p']),
        
        # 最终转换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(transform_list)


class AdvancedCaptchaDataset(torch.utils.data.Dataset):
    """带高级数据增强的验证码数据集 - 继承PyTorch Dataset"""
    def __init__(self, data_dir, augment_strength='medium', transform=None):
        super().__init__()
        
        # 导入基础数据集类
        from .dataset import CaptchaDataset
        
        # 创建基础数据集实例来获取数据和属性
        base_dataset = CaptchaDataset(data_dir, transform=None)
        
        # 继承所有必要属性
        self.characters = base_dataset.characters
        self.char_to_idx = base_dataset.char_to_idx
        self.idx_to_char = base_dataset.idx_to_char
        self.num_classes = base_dataset.num_classes
        self.samples = base_dataset.samples
        
        # 设置高级变换
        if transform is None:
            self.transform = create_advanced_transforms(
                img_height=60, 
                img_width=160, 
                is_train=True, 
                augment_strength=augment_strength
            )
        else:
            self.transform = transform
        
        print(f"高级数据集加载完成: {len(self.samples)} 个样本, 增强强度: {augment_strength}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        image_path, text = self.samples[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用高级变换
        if self.transform:
            image = self.transform(image)
        
        # 转换文本为索引
        text_indices = [self.char_to_idx[char] for char in text]
        text_length = len(text_indices)
        
        return image, torch.tensor(text_indices, dtype=torch.long), text_length, text
    
    def decode_prediction(self, indices):
        """解码预测结果"""
        chars = []
        prev_idx = 0
        
        for idx in indices:
            # CTC解码：跳过blank和重复字符
            if idx != 0 and idx != prev_idx:
                if idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])
            prev_idx = idx
        
        return ''.join(chars)