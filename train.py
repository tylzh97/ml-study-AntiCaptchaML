#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from models import CRNN, CRNNLoss, BasicCRNN, ResNetCRNN
from utils import create_data_loaders
from utils.advanced_dataset import create_advanced_data_loaders


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir='./checkpoints'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, targets, target_lengths, texts) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # 前向传播
            log_probs = self.model(images)
            seq_len, batch_size = log_probs.size(0), log_probs.size(1)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(self.device)
            
            # 计算损失
            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, dataset):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_chars = 0
        total_chars = 0
        correct_sequences = 0
        total_sequences = 0
        
        with torch.no_grad():
            for images, targets, target_lengths, texts in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # 前向传播
                log_probs = self.model(images)
                seq_len, batch_size = log_probs.size(0), log_probs.size(1)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(self.device)
                
                # 计算损失
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.argmax(log_probs, dim=2)  # (seq_len, batch)
                predictions = predictions.permute(1, 0)  # (batch, seq_len)
                
                for i in range(batch_size):
                    pred_text = dataset.decode_prediction(predictions[i].cpu().numpy())
                    true_text = texts[i]
                    
                    # 序列准确率
                    if pred_text == true_text:
                        correct_sequences += 1
                    total_sequences += 1
                    
                    # 字符准确率
                    for pred_char, true_char in zip(pred_text, true_text):
                        if pred_char == true_char:
                            correct_chars += 1
                    total_chars += len(true_text)
        
        avg_loss = total_loss / len(val_loader)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(seq_accuracy)
        
        return avg_loss, char_accuracy, seq_accuracy
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False, save_format='pth'):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        # 根据格式选择文件扩展名
        ext = 'safetensors' if save_format == 'safetensors' else 'pth'
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, f'latest.{ext}')
        self._save_checkpoint_file(checkpoint, latest_path, save_format)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, f'best.{ext}')
            self._save_checkpoint_file(checkpoint, best_path, save_format)
            print(f"保存最佳模型: val_acc={val_acc:.4f} (格式: {save_format})")
    
    def _save_checkpoint_file(self, checkpoint, filepath, save_format):
        """保存检查点文件"""
        if save_format == 'safetensors':
            try:
                from safetensors.torch import save_file
                
                # SafeTensor只能保存张量，需要分别保存
                # 1. 保存模型权重为safetensors
                model_path = filepath
                save_file(checkpoint['model_state_dict'], model_path)
                
                # 2. 保存其他元数据为json
                import json
                metadata_path = filepath.replace('.safetensors', '_metadata.json')
                metadata = {
                    'epoch': checkpoint['epoch'],
                    'val_loss': checkpoint['val_loss'],
                    'val_acc': checkpoint['val_acc'],
                    'train_losses': checkpoint['train_losses'],
                    'val_losses': checkpoint['val_losses'],
                    'val_accuracies': checkpoint['val_accuracies'],
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # 3. 保存优化器状态为pth（safetensors不支持复杂对象）
                optimizer_path = filepath.replace('.safetensors', '_optimizer.pth')
                torch.save({
                    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
                    'scheduler_state_dict': checkpoint['scheduler_state_dict'],
                }, optimizer_path)
                
                print(f"✓ SafeTensor格式保存: {model_path}")
                print(f"  - 模型权重: {model_path}")
                print(f"  - 训练元数据: {metadata_path}")
                print(f"  - 优化器状态: {optimizer_path}")
                
            except ImportError:
                print("⚠️ safetensors库未安装，回退到PyTorch格式")
                torch.save(checkpoint, filepath.replace('.safetensors', '.pth'))
        else:
            # 传统PyTorch格式
            torch.save(checkpoint, filepath)
    
    def train(self, train_loader, val_loader, dataset, num_epochs, save_every=5, save_format='pth'):
        """完整训练循环"""
        best_val_acc = 0.0
        
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"保存格式: {save_format}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, char_acc, seq_acc = self.validate(val_loader, dataset)
            
            # 学习率调度
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 打印结果
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  字符准确率: {char_acc:.4f}")
            print(f"  序列准确率: {seq_acc:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  用时: {epoch_time:.1f}s")
            print("-" * 50)
            
            # 保存检查点
            is_best = seq_acc > best_val_acc
            if is_best:
                best_val_acc = seq_acc
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, seq_acc, is_best, save_format)
        
        print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description='训练CRNN验证码识别模型')
    parser.add_argument('--data_root', type=str, default='./data', 
                        help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='学习率')
    parser.add_argument('--hidden_size', type=int, default=256, 
                        help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='LSTM层数')
    parser.add_argument('--img_height', type=int, default=60, 
                        help='图像高度')
    parser.add_argument('--img_width', type=int, default=160, 
                        help='图像宽度')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载线程数')
    parser.add_argument('--resume', type=str, default='', 
                        help='恢复训练的检查点路径')
    parser.add_argument('--backbone', type=str, default='resnet', 
                        choices=['basic', 'resnet'], help='CNN骨干网络类型')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重 (仅对resnet有效)')
    parser.add_argument('--freeze_backbone', type=str, default='early',
                        choices=['none', 'all', 'early', 'partial'],
                        help='ResNet骨干网络冻结策略 (仅对resnet有效)')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                        help='骨干网络相对学习率比例 (仅对resnet有效)')
    parser.add_argument('--save_format', type=str, default='pth',
                        choices=['pth', 'safetensors'], 
                        help='模型保存格式')
    parser.add_argument('--use_advanced_augment', action='store_true',
                        help='使用高级数据增强（透视变换、弹性形变等）')
    parser.add_argument('--augment_strength', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='数据增强强度')
    
    # 学习率调度参数
    parser.add_argument('--step_size', type=int, default=10,
                        help='学习率衰减步长（每多少个epoch衰减一次）')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='学习率衰减因子')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载器
    if args.use_advanced_augment:
        print(f"🚀 使用高级数据增强，强度: {args.augment_strength}")
        train_loader, val_loader, test_loader, train_dataset = create_advanced_data_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_height=args.img_height,
            img_width=args.img_width,
            augment_strength=args.augment_strength
        )
    else:
        print("📊 使用基础数据增强")
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_height=args.img_height,
            img_width=args.img_width
        )
    
    # 模型
    if args.backbone == 'basic':
        model = BasicCRNN(
            img_height=args.img_height,
            img_width=args.img_width,
            num_classes=train_dataset.num_classes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
        print("使用基础CNN架构")
    else:  # resnet
        model = ResNetCRNN(
            img_height=args.img_height,
            img_width=args.img_width,
            num_classes=train_dataset.num_classes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            pretrained=args.pretrained
        )
        print(f"使用ResNet50架构，预训练权重: {args.pretrained}")
        
        # 应用冻结策略
        model.freeze_backbone(args.freeze_backbone)
    
    model.to(device)
    
    # 损失函数
    criterion = CRNNLoss()
    
    # 优化器 - 对ResNet使用不同学习率
    if args.backbone == 'resnet' and args.freeze_backbone != 'all':
        # 获取参数组
        param_groups = model.get_param_groups(args.backbone_lr_ratio)
        optimizer_params = []
        for group in param_groups:
            optimizer_params.append({
                'params': group['params'],
                'lr': args.lr * group['lr_ratio']
            })
        optimizer = optim.Adam(optimizer_params, weight_decay=1e-4)
        print(f"使用差分学习率: 主干网络lr={args.lr * args.backbone_lr_ratio:.6f}, 其他部分lr={args.lr:.6f}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"使用统一学习率: lr={args.lr:.6f}")
    
    # 学习率调度器 - 更频繁的衰减
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f"学习率调度: 每{args.step_size}个epoch衰减至{args.gamma}倍")
    
    # 训练器
    trainer = Trainer(model, criterion, optimizer, scheduler, device, args.save_dir)
    
    # 恢复训练
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.val_accuracies = checkpoint.get('val_accuracies', [])
    
    # 开始训练
    trainer.train(train_loader, val_loader, train_dataset, args.num_epochs, save_format=args.save_format)


if __name__ == "__main__":
    main()