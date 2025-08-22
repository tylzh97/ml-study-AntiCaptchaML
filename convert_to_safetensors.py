#!/usr/bin/env python3
"""
PyTorch模型转SafeTensors格式工具
"""
import os
import argparse
import torch
import json
from pathlib import Path


def convert_pytorch_to_safetensors(pytorch_path, output_dir=None):
    """
    将PyTorch格式模型转换为SafeTensors格式
    
    Args:
        pytorch_path: PyTorch模型路径 (.pth文件)
        output_dir: 输出目录，默认与输入文件同目录
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("❌ safetensors库未安装，请安装: pip install safetensors")
        return False
    
    pytorch_path = Path(pytorch_path)
    if not pytorch_path.exists():
        print(f"❌ 文件不存在: {pytorch_path}")
        return False
    
    # 设置输出路径
    if output_dir is None:
        output_dir = pytorch_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = pytorch_path.stem
    
    # 输出文件路径
    safetensor_path = output_dir / f"{base_name}.safetensors"
    metadata_path = output_dir / f"{base_name}_metadata.json"
    optimizer_path = output_dir / f"{base_name}_optimizer.pth"
    
    print(f"📥 加载PyTorch模型: {pytorch_path}")
    
    try:
        # 加载PyTorch检查点
        checkpoint = torch.load(pytorch_path, map_location='cpu')
        
        if 'model_state_dict' not in checkpoint:
            print("❌ 检查点文件格式错误，缺少 model_state_dict")
            return False
        
        # 1. 保存模型权重为safetensors
        print(f"💾 保存模型权重: {safetensor_path}")
        save_file(checkpoint['model_state_dict'], str(safetensor_path))
        
        # 2. 保存元数据为JSON
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'val_acc': checkpoint.get('val_acc', 0.0),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'val_accuracies': checkpoint.get('val_accuracies', [])
        }
        
        print(f"📋 保存训练元数据: {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 3. 保存优化器状态为PyTorch格式
        optimizer_data = {}
        if 'optimizer_state_dict' in checkpoint:
            optimizer_data['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        if 'scheduler_state_dict' in checkpoint:
            optimizer_data['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
        
        if optimizer_data:
            print(f"⚙️ 保存优化器状态: {optimizer_path}")
            torch.save(optimizer_data, optimizer_path)
        
        # 验证文件大小
        original_size = pytorch_path.stat().st_size
        safetensor_size = safetensor_path.stat().st_size
        
        print(f"\n✅ 转换完成!")
        print(f"📊 文件大小对比:")
        print(f"  原始文件: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        print(f"  SafeTensor: {safetensor_size:,} bytes ({safetensor_size/1024/1024:.1f} MB)")
        print(f"  压缩比: {safetensor_size/original_size:.2%}")
        
        print(f"\n📁 输出文件:")
        print(f"  模型权重: {safetensor_path}")
        print(f"  训练元数据: {metadata_path}")
        if optimizer_data:
            print(f"  优化器状态: {optimizer_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


def batch_convert(input_dir, output_dir=None):
    """批量转换目录中的所有.pth文件"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    pth_files = list(input_dir.glob("*.pth"))
    if not pth_files:
        print(f"❌ 在 {input_dir} 中未找到.pth文件")
        return
    
    print(f"🔍 发现 {len(pth_files)} 个PyTorch模型文件")
    
    success_count = 0
    for pth_file in pth_files:
        print(f"\n{'='*50}")
        if convert_pytorch_to_safetensors(pth_file, output_dir):
            success_count += 1
    
    print(f"\n🎉 批量转换完成: {success_count}/{len(pth_files)} 成功")


def main():
    parser = argparse.ArgumentParser(description='PyTorch模型转SafeTensors格式工具')
    parser.add_argument('input_path', type=str, 
                        help='输入文件路径(.pth)或目录')
    parser.add_argument('--output_dir', type=str, 
                        help='输出目录，默认与输入文件同目录')
    parser.add_argument('--batch', action='store_true',
                        help='批量转换模式（输入路径为目录）')
    
    args = parser.parse_args()
    
    print("🔄 PyTorch → SafeTensors 转换工具")
    print("=" * 50)
    
    if args.batch:
        batch_convert(args.input_path, args.output_dir)
    else:
        convert_pytorch_to_safetensors(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()