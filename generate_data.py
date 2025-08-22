#!/usr/bin/env python3
import os
import random
import string
from captcha.image import ImageCaptcha
from PIL import Image
import argparse


class CaptchaGenerator:
    def __init__(self, width=160, height=60):
        self.width = width
        self.height = height
        self.image_captcha = ImageCaptcha(width=width, height=height)
        
        # 字符集：数字+大写字母+小写字母
        self.characters = string.digits + string.ascii_letters
        print(f"字符集大小: {len(self.characters)}, 字符: {self.characters}")
    
    def generate_random_text(self, min_len=4, max_len=8):
        """生成随机长度的验证码文本"""
        length = random.randint(min_len, max_len)
        return ''.join(random.choices(self.characters, k=length))
    
    def generate_captcha(self, text):
        """生成验证码图片"""
        data = self.image_captcha.generate(text)
        return Image.open(data)
    
    def generate_dataset(self, output_dir, num_samples, split_ratio=(0.8, 0.1, 0.1)):
        """生成数据集"""
        train_ratio, val_ratio, test_ratio = split_ratio
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "分割比例之和必须为1"
        
        # 创建目录
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        
        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 计算各个分割的样本数
        train_samples = int(num_samples * train_ratio)
        val_samples = int(num_samples * val_ratio)
        test_samples = num_samples - train_samples - val_samples
        
        print(f"生成数据集: 训练{train_samples}, 验证{val_samples}, 测试{test_samples}")
        
        # 生成各个分割的数据
        self._generate_split(train_dir, train_samples, 'train')
        self._generate_split(val_dir, val_samples, 'val')
        self._generate_split(test_dir, test_samples, 'test')
        
        print("数据集生成完成!")
    
    def _generate_split(self, output_dir, num_samples, split_name):
        """生成特定分割的数据"""
        labels_file = os.path.join(output_dir, 'labels.txt')
        
        with open(labels_file, 'w') as f:
            for i in range(num_samples):
                text = self.generate_random_text()
                image = self.generate_captcha(text)
                
                # 保存图片
                image_name = f"{split_name}_{i:06d}.png"
                image_path = os.path.join(output_dir, image_name)
                image.save(image_path)
                
                # 写入标签文件
                f.write(f"{image_name}\t{text}\n")
                
                if (i + 1) % 1000 == 0:
                    print(f"已生成 {split_name} {i + 1}/{num_samples} 样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成验证码数据集')
    parser.add_argument('--output_dir', type=str, default='./data', 
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=50000, 
                        help='总样本数')
    parser.add_argument('--width', type=int, default=160, 
                        help='图片宽度')
    parser.add_argument('--height', type=int, default=60, 
                        help='图片高度')
    
    args = parser.parse_args()
    
    generator = CaptchaGenerator(width=args.width, height=args.height)
    generator.generate_dataset(args.output_dir, args.num_samples)