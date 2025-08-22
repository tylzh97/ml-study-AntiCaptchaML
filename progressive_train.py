#!/usr/bin/env python3
"""
渐进式训练脚本 - 在预训练模型基础上用困难数据继续训练
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path

from models import CRNN, CRNNLoss, BasicCRNN, ResNetCRNN
from utils import create_data_loaders
from utils.advanced_dataset import create_advanced_data_loaders
from train import Trainer


class ProgressiveTrainer(Trainer):
    """渐进式训练器 - 继承基础训练器"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir='./checkpoints'):
        super().__init__(model, criterion, optimizer, scheduler, device, save_dir)
        self.original_performance = {}
        self.hard_performance = {}
    
    def evaluate_on_both_datasets(self, easy_loader, hard_loader, dataset):
        """在简单和困难数据集上都进行评估"""
        print("📊 评估原始数据集性能...")
        easy_loss, easy_char_acc, easy_seq_acc = self.validate(easy_loader, dataset)
        
        print("🔥 评估困难数据集性能...")
        hard_loss, hard_char_acc, hard_seq_acc = self.validate(hard_loader, dataset)
        
        return {
            'easy': {'loss': easy_loss, 'char_acc': easy_char_acc, 'seq_acc': easy_seq_acc},
            'hard': {'loss': hard_loss, 'char_acc': hard_char_acc, 'seq_acc': hard_seq_acc}
        }
    
    def progressive_train(self, hard_train_loader, hard_val_loader, 
                         easy_val_loader, dataset, num_epochs, 
                         save_format='pth', warmup_epochs=5):
        """
        渐进式训练
        
        Args:
            hard_train_loader: 困难数据训练加载器
            hard_val_loader: 困难数据验证加载器  
            easy_val_loader: 原始数据验证加载器
            dataset: 数据集实例
            num_epochs: 训练轮数
            save_format: 保存格式
            warmup_epochs: 热身训练轮数
        """
        print(f"🚀 开始渐进式训练，共{num_epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"热身训练: {warmup_epochs} epochs")
        print(f"保存格式: {save_format}")
        
        best_easy_acc = 0.0
        best_hard_acc = 0.0
        best_overall_acc = 0.0
        
        # 初始评估
        print("\n🔍 训练前性能评估:")
        initial_perf = self.evaluate_on_both_datasets(easy_val_loader, hard_val_loader, dataset)
        print(f"原始数据集 - 序列准确率: {initial_perf['easy']['seq_acc']:.4f}")
        print(f"困难数据集 - 序列准确率: {initial_perf['hard']['seq_acc']:.4f}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 动态调整学习率策略
            if epoch <= warmup_epochs:
                # 热身阶段：使用更小的学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print(f"🔥 热身训练阶段 ({epoch}/{warmup_epochs})")
            elif epoch == warmup_epochs + 1:
                # 恢复正常学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 10
                print("✅ 热身完成，恢复正常学习率")
            
            # 训练一个epoch
            train_loss = self.train_epoch(hard_train_loader, epoch)
            
            # 双重验证
            performance = self.evaluate_on_both_datasets(easy_val_loader, hard_val_loader, dataset)
            easy_seq_acc = performance['easy']['seq_acc']
            hard_seq_acc = performance['hard']['seq_acc']
            
            # 计算综合分数（原始数据权重更高，避免灾难性遗忘）
            overall_score = 0.7 * easy_seq_acc + 0.3 * hard_seq_acc
            
            # 学习率调度
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(overall_score)
            else:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 打印结果
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  原始数据集 - 损失: {performance['easy']['loss']:.4f}, "
                  f"字符准确率: {performance['easy']['char_acc']:.4f}, "
                  f"序列准确率: {easy_seq_acc:.4f}")
            print(f"  困难数据集 - 损失: {performance['hard']['loss']:.4f}, "
                  f"字符准确率: {performance['hard']['char_acc']:.4f}, "
                  f"序列准确率: {hard_seq_acc:.4f}")
            print(f"  综合分数: {overall_score:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  用时: {epoch_time:.1f}s")
            
            # 性能提升检查
            easy_improved = easy_seq_acc > best_easy_acc
            hard_improved = hard_seq_acc > best_hard_acc
            overall_improved = overall_score > best_overall_acc
            
            if easy_improved:
                best_easy_acc = easy_seq_acc
            if hard_improved:
                best_hard_acc = hard_seq_acc
            if overall_improved:
                best_overall_acc = overall_score
            
            # 保存策略
            save_best = False
            if overall_improved:
                save_best = True
                print(f"🎉 综合性能提升! 新最佳分数: {overall_score:.4f}")
            elif easy_seq_acc >= initial_perf['easy']['seq_acc'] * 0.98 and hard_improved:
                # 如果原始性能没有显著下降且困难数据性能提升
                save_best = True
                print(f"🔥 困难数据性能提升且原始性能保持!")
            
            # 保存检查点
            if epoch % 5 == 0 or save_best:
                self.save_checkpoint(epoch, performance['hard']['loss'], overall_score, save_best, save_format)
            
            # 早停检查
            if easy_seq_acc < initial_perf['easy']['seq_acc'] * 0.9:
                print(f"⚠️ 警告: 原始数据性能下降过多 ({easy_seq_acc:.4f} < {initial_perf['easy']['seq_acc']*0.9:.4f})")
                print("考虑降低学习率或提前停止训练")
            
            print("-" * 80)
        
        # 最终报告
        print(f"\n🎯 渐进式训练完成!")
        print(f"初始性能 - 原始: {initial_perf['easy']['seq_acc']:.4f}, 困难: {initial_perf['hard']['seq_acc']:.4f}")
        print(f"最终性能 - 原始: {best_easy_acc:.4f}, 困难: {best_hard_acc:.4f}")
        print(f"最佳综合分数: {best_overall_acc:.4f}")
        
        # 性能提升统计
        easy_gain = best_easy_acc - initial_perf['easy']['seq_acc']
        hard_gain = best_hard_acc - initial_perf['hard']['seq_acc']
        print(f"性能提升 - 原始: {easy_gain:+.4f}, 困难: {hard_gain:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='渐进式训练CRNN验证码识别模型')
    
    # 数据相关
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--easy_data_root', type=str, default='./data',
                        help='原始(简单)数据根目录')
    parser.add_argument('--hard_data_root', type=str, default='./hard_data',
                        help='困难数据根目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='渐进训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='热身训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率(建议比初始训练小)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['basic', 'resnet'], help='CNN骨干网络类型')
    parser.add_argument('--freeze_backbone', type=str, default='partial',
                        choices=['none', 'all', 'early', 'partial'],
                        help='ResNet骨干网络冻结策略')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.05,
                        help='骨干网络学习率比例(建议更小)')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./progressive_checkpoints',
                        help='模型保存目录')
    parser.add_argument('--save_format', type=str, default='safetensors',
                        choices=['pth', 'safetensors'], help='保存格式')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--use_advanced_augment', action='store_true',
                        help='对困难数据使用高级增强')
    
    args = parser.parse_args()
    
    # 检查文件存在
    if not os.path.exists(args.pretrained_model):
        print(f"❌ 预训练模型不存在: {args.pretrained_model}")
        return
    
    if not os.path.exists(args.hard_data_root):
        print(f"❌ 困难数据目录不存在: {args.hard_data_root}")
        print(f"请先运行: python generate_hard_data.py --output_dir {args.hard_data_root}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载原始数据（用于对比评估）
    print("📊 加载原始数据集...")
    easy_train_loader, easy_val_loader, _, easy_dataset = create_data_loaders(
        args.easy_data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 加载困难数据
    print("🔥 加载困难数据集...")
    if args.use_advanced_augment:
        hard_train_loader, hard_val_loader, _, _ = create_advanced_data_loaders(
            args.hard_data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_strength='light'  # 困难数据用轻度增强避免过度
        )
    else:
        hard_train_loader, hard_val_loader, _, _ = create_data_loaders(
            args.hard_data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    # 创建模型
    if args.backbone == 'basic':
        model = BasicCRNN(60, 160, easy_dataset.num_classes)
    else:
        model = ResNetCRNN(60, 160, easy_dataset.num_classes, pretrained=False)
        # 应用冻结策略
        model.freeze_backbone(args.freeze_backbone)
    
    # 加载预训练权重
    print(f"📥 加载预训练模型: {args.pretrained_model}")
    if args.pretrained_model.endswith('.safetensors'):
        from safetensors.torch import load_file
        import json
        
        model_state_dict = load_file(args.pretrained_model)
        metadata_path = args.pretrained_model.replace('.safetensors', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"预训练模型信息 - Epoch: {metadata.get('epoch', 'N/A')}, "
                  f"验证准确率: {metadata.get('val_acc', 'N/A')}")
    else:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        print(f"预训练模型信息 - Epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"验证准确率: {checkpoint.get('val_acc', 'N/A')}")
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    # 设置优化器和调度器
    criterion = CRNNLoss()
    
    if args.backbone == 'resnet' and args.freeze_backbone != 'all':
        # 差分学习率
        param_groups = model.get_param_groups(args.backbone_lr_ratio)
        optimizer_params = []
        for group in param_groups:
            optimizer_params.append({
                'params': group['params'],
                'lr': args.lr * group['lr_ratio']
            })
        optimizer = optim.Adam(optimizer_params, weight_decay=1e-5)  # 更小的权重衰减
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 使用ReduceLROnPlateau来动态调整学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 创建渐进式训练器
    trainer = ProgressiveTrainer(model, criterion, optimizer, scheduler, device, args.save_dir)
    
    # 开始渐进式训练
    trainer.progressive_train(
        hard_train_loader, hard_val_loader, easy_val_loader,
        easy_dataset, args.num_epochs, args.save_format, args.warmup_epochs
    )


if __name__ == "__main__":
    import time
    main()