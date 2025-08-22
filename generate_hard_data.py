#!/usr/bin/env python3
"""
生成困难/畸变验证码数据集 - 用于模型鲁棒性增强
"""
import os
import random
import string
import numpy as np
import cv2
from captcha.image import ImageCaptcha
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import argparse
from pathlib import Path


class HardCaptchaGenerator:
    def __init__(self, width=160, height=60):
        self.width = width
        self.height = height
        self.image_captcha = ImageCaptcha(width=width, height=height)
        self.characters = string.digits + string.ascii_letters
        
    def generate_random_text(self, min_len=4, max_len=8):
        """生成随机长度的验证码文本"""
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(self.characters, k=length))
    
    def add_heavy_distortion(self, image):
        """添加重度畸变"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 1. 透视变换 - 更强烈
        offset = int(min(h, w) * 0.4)  # 增加到40%
        src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        dst_points = np.float32([
            [random.randint(-offset, offset), random.randint(-offset, offset)],
            [w-1 + random.randint(-offset, offset), random.randint(-offset, offset)],
            [w-1 + random.randint(-offset, offset), h-1 + random.randint(-offset, offset)],
            [random.randint(-offset, offset), h-1 + random.randint(-offset, offset)]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_array = cv2.warpPerspective(img_array, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(img_array)
    
    def add_elastic_distortion(self, image):
        """添加弹性变形"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 生成更强的位移场
        alpha = random.randint(30, 50)  # 增强变形强度
        sigma = random.randint(4, 8)
        
        dx = np.random.uniform(-1, 1, (h, w)) * alpha
        dy = np.random.uniform(-1, 1, (h, w)) * alpha
        
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = (x + dx).astype(np.float32)
        y = (y + dy).astype(np.float32)
        
        distorted = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(distorted)
    
    def add_heavy_noise(self, image):
        """添加重度噪声"""
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.3, img_array.shape)
        elif noise_type == 'salt_pepper':
            noise = np.random.random(img_array.shape)
            noise = np.where(noise < 0.1, -1, noise)
            noise = np.where(noise > 0.9, 1, 0)
        else:  # speckle
            noise = np.random.uniform(-0.2, 0.2, img_array.shape)
        
        img_array = np.clip(img_array + noise, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_extreme_blur(self, image):
        """添加极端模糊"""
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            radius = random.uniform(2, 5)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        elif blur_type == 'motion':
            # 运动模糊
            img_array = np.array(image)
            kernel_size = random.randint(9, 15)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = np.ones(kernel_size) / kernel_size
            
            # 随机角度
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
        
        else:  # defocus
            # 散焦模糊
            kernel_size = random.randint(7, 13)
            kernel = np.zeros((kernel_size, kernel_size))
            y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
            mask = x**2 + y**2 <= (kernel_size//2)**2
            kernel[mask] = 1
            kernel = kernel / kernel.sum()
            
            img_array = np.array(image)
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
    
    def add_lighting_variations(self, image):
        """添加光照变化"""
        # 随机亮度和对比度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.3, 1.8))
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.3, 2.0))
        
        return image
    
    def add_interference_lines(self, image):
        """添加干扰线"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # 添加更多干扰线
        num_lines = random.randint(8, 15)
        for _ in range(num_lines):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            width_line = random.randint(1, 3)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width_line)
        
        return image
    
    def add_occlusion(self, image):
        """添加遮挡"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # 随机遮挡块
        num_blocks = random.randint(2, 5)
        for _ in range(num_blocks):
            x1 = random.randint(0, width-20)
            y1 = random.randint(0, height-10)
            x2 = x1 + random.randint(10, 30)
            y2 = y1 + random.randint(5, 15)
            
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([(x1, y1), (x2, y2)], fill=color)
        
        return image
    
    def generate_hard_captcha(self, text, difficulty='hard'):
        """生成困难验证码"""
        # 先生成基础验证码
        data = self.image_captcha.generate(text)
        image = Image.open(data)
        
        # 根据难度级别应用不同强度的畸变
        if difficulty == 'medium':
            # 中等难度：应用部分畸变
            transformations = [
                self.add_heavy_distortion,
                self.add_lighting_variations,
                self.add_interference_lines
            ]
            num_transforms = random.randint(1, 2)
        else:  # hard
            # 高难度：应用多种畸变
            transformations = [
                self.add_heavy_distortion,
                self.add_elastic_distortion,
                self.add_heavy_noise,
                self.add_extreme_blur,
                self.add_lighting_variations,
                self.add_interference_lines,
                self.add_occlusion
            ]
            num_transforms = random.randint(3, 5)
        
        # 随机选择变换
        selected_transforms = random.sample(transformations, num_transforms)
        
        for transform in selected_transforms:
            try:
                image = transform(image)
            except Exception as e:
                print(f"变换失败: {e}")
                continue
        
        return image
    
    def generate_hard_dataset(self, output_dir, num_samples, difficulty='hard'):
        """生成困难数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        labels_file = os.path.join(output_dir, 'labels.txt')
        
        print(f"生成困难数据集: {num_samples} 个样本, 难度: {difficulty}")
        
        with open(labels_file, 'w') as f:
            for i in range(num_samples):
                text = self.generate_random_text()
                image = self.generate_hard_captcha(text, difficulty)
                
                # 保存图片
                image_name = f"hard_{difficulty}_{i:06d}.png"
                image_path = os.path.join(output_dir, image_name)
                image.save(image_path)
                
                # 写入标签
                f.write(f"{image_name}\t{text}\n")
                
                if (i + 1) % 500 == 0:
                    print(f"已生成 {i + 1}/{num_samples} 困难样本")
        
        print(f"困难数据集生成完成: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='生成困难/畸变验证码数据集')
    parser.add_argument('--output_dir', type=str, default='./hard_data',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='生成样本数量')
    parser.add_argument('--difficulty', type=str, choices=['medium', 'hard'], 
                        default='hard', help='困难程度')
    parser.add_argument('--width', type=int, default=160,
                        help='图片宽度')
    parser.add_argument('--height', type=int, default=60,
                        help='图片高度')
    
    args = parser.parse_args()
    
    generator = HardCaptchaGenerator(width=args.width, height=args.height)
    
    # 创建困难数据集目录结构
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    
    # 按8:2分割训练和验证
    train_samples = int(args.num_samples * 0.8)
    val_samples = args.num_samples - train_samples
    
    generator.generate_hard_dataset(train_dir, train_samples, args.difficulty)
    generator.generate_hard_dataset(val_dir, val_samples, args.difficulty)
    
    print(f"\n🎉 困难数据集生成完成!")
    print(f"训练集: {train_samples} 样本")
    print(f"验证集: {val_samples} 样本")
    print(f"总计: {args.num_samples} 样本")
    print(f"难度级别: {args.difficulty}")


if __name__ == "__main__":
    main()