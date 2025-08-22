#!/usr/bin/env python3
import os
import time
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from models import CRNN, CRNNLoss
from utils import create_data_loaders


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
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
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
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: val_acc={val_acc:.4f}")
    
    def train(self, train_loader, val_loader, dataset, num_epochs, save_every=5):
        """完整训练循环"""
        best_val_acc = 0.0
        
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
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
                self.save_checkpoint(epoch, val_loss, seq_acc, is_best)
        
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
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载器
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    # 模型
    model = CRNN(
        img_height=args.img_height,
        img_width=args.img_width,
        num_classes=train_dataset.num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    model.to(device)
    
    # 损失函数
    criterion = CRNNLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
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
    trainer.train(train_loader, val_loader, train_dataset, args.num_epochs)


if __name__ == "__main__":
    main()