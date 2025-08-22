#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import CRNN, BasicCRNN, ResNetCRNN
from utils import create_data_transforms, CaptchaDataset


class CaptchaInference:
    def __init__(self, model_path, device='cuda'):
        """
        验证码推理类
        
        Args:
            model_path: 模型权重路径
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        if model_path.endswith('.safetensors'):
            checkpoint = self._load_safetensor_checkpoint(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        # 尝试自动检测模型架构
        model_state_keys = list(checkpoint['model_state_dict'].keys())
        is_resnet = any('cnn.conv1' in key or 'cnn.0.weight' in key for key in model_state_keys)
        
        # 创建模型实例
        if is_resnet and any('cnn.0.weight' in key for key in model_state_keys):
            print("检测到ResNet架构模型")
            self.model = ResNetCRNN(
                img_height=60,
                img_width=160,
                num_classes=62,
                hidden_size=256,
                num_layers=2,
                pretrained=False  # 推理时不需要预训练权重
            )
        else:
            print("使用基础CRNN架构模型")
            self.model = BasicCRNN(
                img_height=60,
                img_width=160,
                num_classes=62,
                hidden_size=256,
                num_layers=2
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 创建数据变换
        self.transform = create_data_transforms(60, 160, is_train=False)
        
        # 创建字符映射（需要与训练时保持一致）
        import string
        characters = string.digits + string.ascii_letters
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        self.idx_to_char[0] = '<blank>'
        
        print(f"模型加载完成，设备: {self.device}")
        if 'epoch' in checkpoint:
            print(f"模型训练epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"验证准确率: {checkpoint['val_acc']:.4f}")
    
    def _load_safetensor_checkpoint(self, model_path):
        """加载SafeTensor格式的检查点"""
        try:
            from safetensors.torch import load_file
            import json
            
            # 加载模型权重
            model_state_dict = load_file(model_path)
            
            # 加载元数据
            metadata_path = model_path.replace('.safetensors', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                print("⚠️ 未找到元数据文件，使用默认值")
            
            # 构造checkpoint格式
            checkpoint = {
                'model_state_dict': model_state_dict,
                'epoch': metadata.get('epoch', 0),
                'val_loss': metadata.get('val_loss', 0.0),
                'val_acc': metadata.get('val_acc', 0.0),
                'train_losses': metadata.get('train_losses', []),
                'val_losses': metadata.get('val_losses', []),
                'val_accuracies': metadata.get('val_accuracies', [])
            }
            
            print(f"✓ 成功加载SafeTensor模型: {model_path}")
            return checkpoint
            
        except ImportError:
            print("❌ safetensors库未安装，请安装: pip install safetensors")
            raise
        except Exception as e:
            print(f"❌ 加载SafeTensor模型失败: {e}")
            raise
    
    def decode_prediction(self, indices):
        """CTC解码"""
        chars = []
        prev_idx = 0
        
        for idx in indices:
            if idx != 0 and idx != prev_idx:  # 跳过blank和重复
                if idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])
            prev_idx = idx
        
        return ''.join(chars)
    
    def predict_image(self, image_path):
        """预测单张图片"""
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 前向传播
            log_probs = self.model(input_tensor)  # (seq_len, 1, num_classes+1)
            
            # 获取预测结果
            predictions = torch.argmax(log_probs, dim=2).squeeze(1)  # (seq_len,)
            predicted_text = self.decode_prediction(predictions.cpu().numpy())
        
        return predicted_text, log_probs
    
    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        
        for image_path in tqdm(image_paths, desc="批量推理"):
            try:
                predicted_text, _ = self.predict_image(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_text': predicted_text
                })
            except Exception as e:
                print(f"预测失败 {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'predicted_text': ''
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """可视化预测结果"""
        predicted_text, log_probs = self.predict_image(image_path)
        
        # 加载原图
        image = Image.open(image_path)
        
        # 创建图像
        plt.figure(figsize=(12, 4))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"输入图像")
        plt.axis('off')
        
        # 显示预测概率
        plt.subplot(1, 2, 2)
        probs = torch.softmax(log_probs, dim=2).squeeze(1).cpu().numpy()  # (seq_len, num_classes+1)
        plt.imshow(probs.T, aspect='auto', cmap='Blues')
        plt.xlabel('时间步')
        plt.ylabel('字符类别')
        plt.title(f"预测结果: {predicted_text}")
        plt.colorbar()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return predicted_text


def evaluate_model(model_path, data_root, split='test'):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建推理器
    inferencer = CaptchaInference(model_path, device)
    
    # 创建数据集
    test_dir = os.path.join(data_root, split)
    dataset = CaptchaDataset(test_dir, transform=inferencer.transform)
    
    print(f"评估数据集: {len(dataset)} 个样本")
    
    correct_sequences = 0
    correct_chars = 0
    total_chars = 0
    
    results = []
    
    for i in tqdm(range(len(dataset)), desc="评估中"):
        image, _, _, true_text = dataset[i]
        image_path = dataset.samples[i][0]
        
        try:
            predicted_text, _ = inferencer.predict_image(image_path)
            
            # 序列准确率
            is_correct = predicted_text == true_text
            if is_correct:
                correct_sequences += 1
            
            # 字符准确率
            min_len = min(len(predicted_text), len(true_text))
            for j in range(min_len):
                if predicted_text[j] == true_text[j]:
                    correct_chars += 1
            total_chars += len(true_text)
            
            results.append({
                'image_path': image_path,
                'true_text': true_text,
                'predicted_text': predicted_text,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"评估失败 {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'true_text': true_text,
                'predicted_text': '',
                'correct': False
            })
    
    # 计算准确率
    seq_accuracy = correct_sequences / len(dataset)
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    print(f"\n评估结果:")
    print(f"序列准确率: {seq_accuracy:.4f} ({correct_sequences}/{len(dataset)})")
    print(f"字符准确率: {char_accuracy:.4f} ({correct_chars}/{total_chars})")
    
    # 显示一些错误案例
    wrong_cases = [r for r in results if not r['correct']]
    if wrong_cases:
        print(f"\n错误案例 (显示前10个):")
        for i, case in enumerate(wrong_cases[:10]):
            print(f"  {i+1}. 真实: '{case['true_text']}' -> 预测: '{case['predicted_text']}'")
    
    return results, seq_accuracy, char_accuracy


def main():
    parser = argparse.ArgumentParser(description='CRNN验证码识别推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate'], 
                        default='single', help='推理模式')
    parser.add_argument('--image_path', type=str, 
                        help='单张图片路径 (single模式)')
    parser.add_argument('--image_dir', type=str,
                        help='图片目录 (batch模式)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据根目录 (evaluate模式)')
    parser.add_argument('--output_path', type=str,
                        help='结果保存路径')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image_path:
            print("single模式需要指定 --image_path")
            return
        
        inferencer = CaptchaInference(args.model_path)
        
        if args.visualize:
            predicted_text = inferencer.visualize_prediction(
                args.image_path, args.output_path
            )
        else:
            predicted_text, _ = inferencer.predict_image(args.image_path)
        
        print(f"预测结果: {predicted_text}")
        
    elif args.mode == 'batch':
        if not args.image_dir:
            print("batch模式需要指定 --image_dir")
            return
        
        # 获取所有图片路径
        image_paths = []
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        inferencer = CaptchaInference(args.model_path)
        results = inferencer.predict_batch(image_paths)
        
        # 输出结果
        for result in results:
            print(f"{result['image_path']}: {result['predicted_text']}")
        
        # 保存结果
        if args.output_path:
            with open(args.output_path, 'w') as f:
                for result in results:
                    f.write(f"{result['image_path']}\t{result['predicted_text']}\n")
        
    elif args.mode == 'evaluate':
        results, seq_acc, char_acc = evaluate_model(args.model_path, args.data_root)
        
        # 保存评估结果
        if args.output_path:
            with open(args.output_path, 'w') as f:
                f.write(f"序列准确率: {seq_acc:.4f}\n")
                f.write(f"字符准确率: {char_acc:.4f}\n\n")
                f.write("详细结果:\n")
                for result in results:
                    status = "✓" if result['correct'] else "✗"
                    f.write(f"{status} {result['true_text']} -> {result['predicted_text']}\n")


if __name__ == "__main__":
    main()