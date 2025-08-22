#!/usr/bin/env python3
"""
训练问题诊断脚本
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import CRNN, BasicCRNN, ResNetCRNN
from utils import create_data_loaders, CaptchaDataset


def check_data_quality(data_root):
    """检查数据质量"""
    print("🔍 数据质量检查")
    print("=" * 50)
    
    # 检查目录结构
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if os.path.exists(split_dir):
            labels_file = os.path.join(split_dir, 'labels.txt')
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    lines = f.readlines()
                print(f"✅ {split}: {len(lines)} 样本")
                
                # 检查前几个样本
                print(f"   前3个样本:")
                for i, line in enumerate(lines[:3]):
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        img_name, text = parts
                        img_path = os.path.join(split_dir, img_name)
                        exists = "✅" if os.path.exists(img_path) else "❌"
                        print(f"   {exists} {img_name} -> '{text}' (长度:{len(text)})")
                    else:
                        print(f"   ❌ 格式错误: {line.strip()}")
            else:
                print(f"❌ {split}: labels.txt 不存在")
        else:
            print(f"❌ {split}: 目录不存在")


def test_dataloader(data_root):
    """测试数据加载器"""
    print("\n📊 数据加载器测试")
    print("=" * 50)
    
    try:
        train_loader, val_loader, test_loader, dataset = create_data_loaders(
            data_root, batch_size=4, num_workers=0
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   字符集大小: {dataset.num_classes}")
        print(f"   字符集: {dataset.characters}")
        
        # 测试训练数据
        print(f"\n📈 训练数据测试:")
        for i, (images, text_indices, text_lengths, texts) in enumerate(train_loader):
            print(f"   批次 {i+1}:")
            print(f"     图像形状: {images.shape}")
            print(f"     文本: {texts}")
            print(f"     文本长度: {text_lengths.tolist()}")
            print(f"     索引形状: {text_indices.shape}")
            print(f"     索引内容: {text_indices[:20].tolist()}...")
            
            # 检查解码
            start_idx = 0
            for j, length in enumerate(text_lengths):
                indices = text_indices[start_idx:start_idx+length]
                decoded = dataset.decode_prediction(indices.numpy())
                print(f"     解码测试 {j+1}: {texts[j]} -> {decoded}")
                start_idx += length
            
            if i >= 1:  # 只测试前2个批次
                break
                
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_output(data_root):
    """测试模型输出"""
    print("\n🧠 模型输出测试")
    print("=" * 50)
    
    try:
        # 创建数据
        train_loader, val_loader, test_loader, dataset = create_data_loaders(
            data_root, batch_size=2, num_workers=0
        )
        
        # 测试基础CRNN
        print("📊 测试基础CRNN:")
        model = BasicCRNN(60, 160, dataset.num_classes)
        model.eval()
        
        with torch.no_grad():
            for images, text_indices, text_lengths, texts in train_loader:
                print(f"   输入图像: {images.shape}")
                
                # 前向传播
                log_probs = model(images)
                print(f"   模型输出: {log_probs.shape}")
                print(f"   输出格式: (seq_len={log_probs.shape[0]}, batch={log_probs.shape[1]}, classes={log_probs.shape[2]})")
                
                # CTC输入长度
                seq_len, batch_size = log_probs.shape[0], log_probs.shape[1]
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
                print(f"   CTC输入长度: {input_lengths.tolist()}")
                print(f"   目标长度: {text_lengths.tolist()}")
                
                # 解码测试
                predictions = torch.argmax(log_probs, dim=2)  # (seq_len, batch)
                predictions = predictions.permute(1, 0)  # (batch, seq_len)
                
                for i in range(batch_size):
                    pred_text = dataset.decode_prediction(predictions[i].numpy())
                    true_text = texts[i]
                    print(f"   样本 {i+1}: 真实='{true_text}' 预测='{pred_text}'")
                
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_samples(data_root, num_samples=4):
    """可视化样本"""
    print(f"\n🖼️ 样本可视化 (前{num_samples}个)")
    print("=" * 50)
    
    try:
        dataset = CaptchaDataset(os.path.join(data_root, 'train'))
        
        fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(dataset))):
            image, _, _, text = dataset[i]
            
            # 反归一化显示
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # CHW格式
                    image = image.permute(1, 2, 0)
                # 假设已归一化，需要反归一化
                image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                image = torch.clamp(image, 0, 1)
            
            axes[i].imshow(image)
            axes[i].set_title(f"'{text}'")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('debug_samples.png', dpi=150, bbox_inches='tight')
        print("✅ 样本图像已保存到 debug_samples.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return False


def main():
    data_root = "./data"
    
    print("🔧 CRNN训练问题诊断工具")
    print("=" * 60)
    
    # 1. 检查数据质量
    check_data_quality(data_root)
    
    # 2. 测试数据加载
    if not test_dataloader(data_root):
        print("\n❌ 数据加载失败，请检查数据格式")
        return
    
    # 3. 测试模型
    if not test_model_output(data_root):
        print("\n❌ 模型测试失败")
        return
    
    # 4. 可视化样本
    try:
        visualize_samples(data_root)
    except:
        print("⚠️ 样本可视化失败（可能没有matplotlib）")
    
    print("\n🎯 诊断建议:")
    print("1. 如果数据格式正确，尝试降低学习率到1e-4或5e-5")
    print("2. 检查生成的验证码是否过于复杂")
    print("3. 尝试减少验证码长度变化范围")
    print("4. 考虑增加训练数据量")
    print("5. 检查模型输出序列长度是否合理")


if __name__ == "__main__":
    main()