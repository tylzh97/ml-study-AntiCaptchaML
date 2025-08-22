#!/usr/bin/env python3
"""
预训练模型下载脚本 - 支持离线部署
"""
import os
import sys
import urllib.request
import shutil
from pathlib import Path


def download_resnet50():
    """下载ResNet50预训练模型"""
    model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    model_filename = "resnet50-0676ba61.pth"
    
    # 创建本地预训练模型目录
    pretrained_dir = Path("./pretrained")
    pretrained_dir.mkdir(exist_ok=True)
    
    local_path = pretrained_dir / model_filename
    
    if local_path.exists():
        print(f"✓ 模型已存在: {local_path}")
        return str(local_path)
    
    print(f"📥 开始下载ResNet50预训练模型...")
    print(f"URL: {model_url}")
    print(f"保存路径: {local_path}")
    
    try:
        # 下载模型
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100.0 / total_size
                sys.stdout.write(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(model_url, local_path, progress_hook)
        print(f"\n✅ 下载完成: {local_path}")
        
        # 验证文件大小
        file_size = local_path.stat().st_size
        expected_size = 97781949  # ResNet50模型的预期大小(字节)
        
        if abs(file_size - expected_size) < 1000:  # 允许小幅差异
            print(f"✓ 文件大小验证通过: {file_size} bytes")
        else:
            print(f"⚠️ 文件大小异常: {file_size} bytes (期望: {expected_size} bytes)")
        
        return str(local_path)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def setup_torch_cache():
    """设置PyTorch缓存目录"""
    import torch
    
    # 获取PyTorch缓存目录
    torch_cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    cache_model_path = torch_cache_dir / "resnet50-0676ba61.pth"
    local_model_path = Path("./pretrained/resnet50-0676ba61.pth")
    
    if cache_model_path.exists():
        print(f"✓ PyTorch缓存中已存在模型: {cache_model_path}")
        return True
    
    if local_model_path.exists():
        # 复制到缓存目录
        shutil.copy2(local_model_path, cache_model_path)
        print(f"✓ 已复制模型到PyTorch缓存: {cache_model_path}")
        return True
    
    return False


def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载...")
    
    try:
        import torchvision.models as models
        
        # 测试在线加载
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            print("✓ 在线预训练模型加载成功")
            return True
        except:
            print("⚠️ 在线加载失败，尝试本地加载...")
        
        # 测试本地加载
        local_path = "./pretrained/resnet50-0676ba61.pth"
        if os.path.exists(local_path):
            import torch
            model = models.resnet50(weights=None)
            state_dict = torch.load(local_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"✓ 本地预训练模型加载成功: {local_path}")
            return True
        else:
            print(f"❌ 本地模型文件不存在: {local_path}")
            return False
            
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False


def main():
    print("🔧 ResNet50预训练模型部署工具")
    print("=" * 50)
    
    # 1. 下载模型
    model_path = download_resnet50()
    if not model_path:
        print("❌ 下载失败，请检查网络连接")
        return
    
    # 2. 设置PyTorch缓存
    setup_torch_cache()
    
    # 3. 测试加载
    if test_model_loading():
        print("\n🎉 预训练模型部署完成!")
        print("\n📋 使用说明:")
        print("1. 在线环境: 模型会自动从缓存加载")
        print("2. 离线环境: 将 ./pretrained/ 目录拷贝到目标机器")
        print("3. 训练命令: uv run python train.py --backbone resnet --pretrained")
    else:
        print("\n❌ 模型部署失败，请检查错误信息")
    
    print(f"\n📁 文件位置:")
    print(f"  本地模型: ./pretrained/resnet50-0676ba61.pth")
    print(f"  缓存模型: ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")


if __name__ == "__main__":
    main()