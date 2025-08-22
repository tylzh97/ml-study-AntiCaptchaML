#!/usr/bin/env python3
"""
快速开始脚本 - 一键完成数据生成、训练和测试
"""
import os
import subprocess
import argparse


def run_command(cmd, description):
    """运行命令并打印进度"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} 完成")
        if result.stdout:
            print("输出:")
            print(result.stdout)
    else:
        print(f"✗ {description} 失败")
        print("错误信息:")
        print(result.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='CRNN验证码识别快速开始')
    parser.add_argument('--num_samples', type=int, default=10000, 
                        help='生成样本数量')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--skip_data_gen', action='store_true', 
                        help='跳过数据生成')
    parser.add_argument('--skip_training', action='store_true', 
                        help='跳过训练')
    
    args = parser.parse_args()
    
    print("CRNN验证码识别 - 快速开始")
    print(f"样本数量: {args.num_samples}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    
    # 1. 生成数据
    if not args.skip_data_gen:
        if not run_command(
            f"python generate_data.py --num_samples {args.num_samples} --output_dir ./data",
            "生成训练数据"
        ):
            return
    else:
        print("跳过数据生成")
    
    # 2. 训练模型
    if not args.skip_training:
        if not run_command(
            f"python train.py --data_root ./data --num_epochs {args.num_epochs} --batch_size {args.batch_size} --save_dir ./checkpoints",
            "训练CRNN模型"
        ):
            return
    else:
        print("跳过模型训练")
    
    # 3. 评估模型
    best_model_path = "./checkpoints/best.pth"
    if os.path.exists(best_model_path):
        if not run_command(
            f"python inference.py --model_path {best_model_path} --mode evaluate --data_root ./data --output_path ./output/evaluation_results.txt",
            "评估模型性能"
        ):
            return
    else:
        print("未找到最佳模型权重，跳过评估")
    
    print(f"\n{'='*50}")
    print("🎉 快速开始完成!")
    print("接下来你可以:")
    print("1. 查看训练日志和模型权重: ./checkpoints/")
    print("2. 查看评估结果: ./output/evaluation_results.txt")
    print("3. 使用inference.py进行单张图片预测")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()