#!/usr/bin/env python3
"""
模型微调便捷工具 - 一键完成困难数据生成和渐进式训练
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    if check:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ {description} 失败!")
            return False
        print(f"✅ {description} 完成!")
    else:
        subprocess.run(cmd, shell=True)
    
    return True


def finetune_pipeline(args):
    """完整的微调流程"""
    
    print("🚀 CRNN模型渐进式微调流程")
    print("=" * 60)
    print(f"预训练模型: {args.pretrained_model}")
    print(f"困难数据样本数: {args.hard_samples}")
    print(f"困难程度: {args.difficulty}")
    print(f"微调轮数: {args.finetune_epochs}")
    
    # 检查预训练模型
    if not os.path.exists(args.pretrained_model):
        print(f"❌ 预训练模型不存在: {args.pretrained_model}")
        return False
    
    # 1. 生成困难数据
    if not args.skip_data_generation:
        hard_data_cmd = f"""python generate_hard_data.py \
            --output_dir {args.hard_data_dir} \
            --num_samples {args.hard_samples} \
            --difficulty {args.difficulty} \
            --width 160 --height 60"""
        
        if not run_command(hard_data_cmd, "生成困难验证码数据"):
            return False
    else:
        print("⏭️ 跳过数据生成")
        if not os.path.exists(args.hard_data_dir):
            print(f"❌ 困难数据目录不存在且跳过生成: {args.hard_data_dir}")
            return False
    
    # 2. 渐进式训练
    progressive_cmd = f"""python progressive_train.py \
        --pretrained_model {args.pretrained_model} \
        --easy_data_root {args.easy_data_dir} \
        --hard_data_root {args.hard_data_dir} \
        --num_epochs {args.finetune_epochs} \
        --warmup_epochs {args.warmup_epochs} \
        --lr {args.learning_rate} \
        --batch_size {args.batch_size} \
        --backbone {args.backbone} \
        --freeze_backbone {args.freeze_strategy} \
        --save_dir {args.output_dir} \
        --save_format {args.save_format}"""
    
    if args.use_advanced_augment:
        progressive_cmd += " --use_advanced_augment"
    
    if not run_command(progressive_cmd, "渐进式模型训练"):
        return False
    
    # 3. 模型评估
    best_model = os.path.join(args.output_dir, f"best.{args.save_format}")
    if os.path.exists(best_model):
        print(f"\n🎯 微调完成! 最佳模型: {best_model}")
        
        # 在原始测试集上评估
        eval_cmd = f"""python inference.py \
            --model_path {best_model} \
            --mode evaluate \
            --data_root {args.easy_data_dir} \
            --output_path {os.path.join(args.output_dir, 'easy_eval.txt')}"""
        
        print("\n📊 在原始测试集上评估性能:")
        run_command(eval_cmd, "原始数据评估", check=False)
        
        # 在困难测试集上评估
        if os.path.exists(os.path.join(args.hard_data_dir, 'test')):
            eval_hard_cmd = f"""python inference.py \
                --model_path {best_model} \
                --mode evaluate \
                --data_root {args.hard_data_dir} \
                --output_path {os.path.join(args.output_dir, 'hard_eval.txt')}"""
            
            print("\n🔥 在困难测试集上评估性能:")
            run_command(eval_hard_cmd, "困难数据评估", check=False)
        
        print(f"\n🎉 微调流程完成!")
        print(f"📁 输出目录: {args.output_dir}")
        print(f"🏆 最佳模型: {best_model}")
        return True
    else:
        print("❌ 未找到最佳模型文件")
        return False


def main():
    parser = argparse.ArgumentParser(description='CRNN模型渐进式微调工具')
    
    # 必需参数
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='预训练模型路径')
    
    # 数据相关
    parser.add_argument('--easy_data_dir', type=str, default='./data',
                        help='原始数据目录')
    parser.add_argument('--hard_data_dir', type=str, default='./hard_data',
                        help='困难数据目录')
    parser.add_argument('--output_dir', type=str, default='./finetuned_models',
                        help='微调模型输出目录')
    
    # 困难数据生成
    parser.add_argument('--hard_samples', type=int, default=10000,
                        help='生成困难样本数量')
    parser.add_argument('--difficulty', type=str, choices=['medium', 'hard'],
                        default='hard', help='困难程度')
    parser.add_argument('--skip_data_generation', action='store_true',
                        help='跳过数据生成(使用现有困难数据)')
    
    # 训练参数
    parser.add_argument('--finetune_epochs', type=int, default=25,
                        help='微调训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='热身轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='微调学习率(建议比初训更小)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['basic', 'resnet'], help='骨干网络')
    parser.add_argument('--freeze_strategy', type=str, default='partial',
                        choices=['none', 'all', 'early', 'partial'],
                        help='冻结策略')
    parser.add_argument('--save_format', type=str, default='safetensors',
                        choices=['pth', 'safetensors'], help='保存格式')
    
    # 高级选项
    parser.add_argument('--use_advanced_augment', action='store_true',
                        help='使用高级数据增强')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行微调流程
    success = finetune_pipeline(args)
    
    if success:
        print("\n🎊 微调完成! 建议接下来:")
        print("1. 比较微调前后在原始测试集上的性能")
        print("2. 测试模型在实际困难验证码上的表现")
        print("3. 如果性能满意，可以进一步生成更困难的数据继续微调")
        sys.exit(0)
    else:
        print("\n💥 微调失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()