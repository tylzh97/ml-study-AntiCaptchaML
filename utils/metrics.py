import torch
import numpy as np
from collections import Counter


def calculate_accuracy(predictions, targets, dataset):
    """
    计算预测准确率
    
    Args:
        predictions: 模型预测结果 (batch, seq_len)
        targets: 真实标签列表
        dataset: 数据集实例，用于解码
    
    Returns:
        char_accuracy: 字符准确率
        seq_accuracy: 序列准确率
    """
    batch_size = predictions.size(0)
    correct_chars = 0
    total_chars = 0
    correct_sequences = 0
    
    for i in range(batch_size):
        # 解码预测结果
        pred_text = dataset.decode_prediction(predictions[i].cpu().numpy())
        true_text = targets[i]
        
        # 序列准确率
        if pred_text == true_text:
            correct_sequences += 1
        
        # 字符准确率
        min_len = min(len(pred_text), len(true_text))
        for j in range(min_len):
            if pred_text[j] == true_text[j]:
                correct_chars += 1
        total_chars += len(true_text)
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    seq_accuracy = correct_sequences / batch_size
    
    return char_accuracy, seq_accuracy


def analyze_errors(predictions, targets, dataset):
    """
    分析预测错误
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        dataset: 数据集实例
    
    Returns:
        error_analysis: 错误分析结果
    """
    char_errors = Counter()  # 字符错误统计
    length_errors = Counter()  # 长度错误统计
    
    batch_size = predictions.size(0)
    error_cases = []
    
    for i in range(batch_size):
        pred_text = dataset.decode_prediction(predictions[i].cpu().numpy())
        true_text = targets[i]
        
        if pred_text != true_text:
            error_cases.append({
                'predicted': pred_text,
                'true': true_text,
                'pred_len': len(pred_text),
                'true_len': len(true_text)
            })
            
            # 长度错误统计
            len_diff = len(pred_text) - len(true_text)
            length_errors[len_diff] += 1
            
            # 字符错误统计
            min_len = min(len(pred_text), len(true_text))
            for j in range(min_len):
                if pred_text[j] != true_text[j]:
                    char_errors[f"{true_text[j]}->{pred_text[j]}"] += 1
    
    return {
        'error_cases': error_cases,
        'char_errors': dict(char_errors),
        'length_errors': dict(length_errors)
    }


class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.total_loss = 0.0
        self.correct_chars = 0
        self.total_chars = 0
        self.correct_sequences = 0
        self.total_sequences = 0
        self.num_batches = 0
    
    def update(self, loss, predictions, targets, dataset):
        """更新指标"""
        self.total_loss += loss
        self.num_batches += 1
        
        batch_size = predictions.size(0)
        self.total_sequences += batch_size
        
        for i in range(batch_size):
            pred_text = dataset.decode_prediction(predictions[i].cpu().numpy())
            true_text = targets[i]
            
            # 序列准确率
            if pred_text == true_text:
                self.correct_sequences += 1
            
            # 字符准确率
            min_len = min(len(pred_text), len(true_text))
            for j in range(min_len):
                if pred_text[j] == true_text[j]:
                    self.correct_chars += 1
            self.total_chars += len(true_text)
    
    def get_metrics(self):
        """获取当前指标"""
        avg_loss = self.total_loss / self.num_batches if self.num_batches > 0 else 0
        char_acc = self.correct_chars / self.total_chars if self.total_chars > 0 else 0
        seq_acc = self.correct_sequences / self.total_sequences if self.total_sequences > 0 else 0
        
        return {
            'avg_loss': avg_loss,
            'char_accuracy': char_acc,
            'seq_accuracy': seq_acc,
            'num_samples': self.total_sequences
        }


def edit_distance(s1, s2):
    """
    计算编辑距离（Levenshtein距离）
    
    Args:
        s1, s2: 两个字符串
    
    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_edit_distance_accuracy(predictions, targets, dataset):
    """
    基于编辑距离计算准确率
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        dataset: 数据集实例
    
    Returns:
        平均编辑距离和归一化准确率
    """
    batch_size = predictions.size(0)
    total_edit_distance = 0
    total_length = 0
    
    for i in range(batch_size):
        pred_text = dataset.decode_prediction(predictions[i].cpu().numpy())
        true_text = targets[i]
        
        ed = edit_distance(pred_text, true_text)
        total_edit_distance += ed
        total_length += len(true_text)
    
    avg_edit_distance = total_edit_distance / batch_size
    normalized_accuracy = 1 - (total_edit_distance / total_length) if total_length > 0 else 0
    
    return avg_edit_distance, normalized_accuracy