import os
import torch
from torch.utils.data import DataLoader
from .dataset import CaptchaDataset, collate_fn
from .advanced_transforms import AdvancedCaptchaDataset, create_advanced_transforms


def create_advanced_data_loaders(data_root, batch_size=32, num_workers=4, 
                                img_height=60, img_width=160, 
                                augment_strength='medium'):
    """
    创建带高级数据增强的数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_height: 图像高度
        img_width: 图像宽度
        augment_strength: 增强强度 ('light', 'medium', 'heavy')
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset
    """
    
    # 创建变换
    train_transform = create_advanced_transforms(
        img_height, img_width, is_train=True, augment_strength=augment_strength
    )
    val_transform = create_advanced_transforms(
        img_height, img_width, is_train=False
    )
    
    # 创建数据集
    train_dataset = AdvancedCaptchaDataset(
        os.path.join(data_root, 'train'),
        augment_strength=augment_strength,
        transform=train_transform
    )
    
    val_dataset = AdvancedCaptchaDataset(
        os.path.join(data_root, 'val'),
        augment_strength='light',  # 验证集用轻度增强
        transform=val_transform
    )
    
    test_dataset = AdvancedCaptchaDataset(
        os.path.join(data_root, 'test'),
        augment_strength=None,  # 测试集不增强
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def preview_augmentations(data_root, num_samples=8, augment_strength='medium'):
    """
    预览数据增强效果
    
    Args:
        data_root: 数据根目录
        num_samples: 预览样本数
        augment_strength: 增强强度
    """
    import matplotlib.pyplot as plt
    from .dataset import CaptchaDataset
    
    # 创建原始数据集和增强数据集
    original_dataset = CaptchaDataset(
        os.path.join(data_root, 'train'),
        transform=create_advanced_transforms(60, 160, is_train=False)
    )
    
    augmented_dataset = AdvancedCaptchaDataset(
        os.path.join(data_root, 'train'),
        augment_strength=augment_strength
    )
    
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 6))
    fig.suptitle(f'数据增强预览 (强度: {augment_strength})', fontsize=14)
    
    for i in range(num_samples):
        # 原始图像
        orig_img, _, _, orig_text = original_dataset[i]
        orig_img = orig_img.permute(1, 2, 0)  # CHW -> HWC
        # 反归一化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for c in range(3):
            orig_img[:, :, c] = orig_img[:, :, c] * std[c] + mean[c]
        orig_img = torch.clamp(orig_img, 0, 1)
        
        # 增强图像
        aug_img, _, _, aug_text = augmented_dataset[i]
        aug_img = aug_img.permute(1, 2, 0)
        for c in range(3):
            aug_img[:, :, c] = aug_img[:, :, c] * std[c] + mean[c]
        aug_img = torch.clamp(aug_img, 0, 1)
        
        # 显示
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'原始: {orig_text}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_img)
        axes[1, i].set_title(f'增强: {aug_text}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch
    
    # 测试高级数据加载器
    data_root = "./data"
    
    if os.path.exists(data_root):
        print("🧪 测试高级数据增强加载器...")
        
        # 创建不同强度的数据加载器
        for strength in ['light', 'medium', 'heavy']:
            print(f"\n📊 测试增强强度: {strength}")
            
            train_loader, val_loader, test_loader, train_dataset = create_advanced_data_loaders(
                data_root, batch_size=4, augment_strength=strength
            )
            
            # 测试一个批次
            for batch_idx, (images, text_indices, text_lengths, texts) in enumerate(train_loader):
                print(f"  批次 {batch_idx}:")
                print(f"    图像形状: {images.shape}")
                print(f"    文本: {texts}")
                break
        
        print("\n✅ 高级数据加载器测试完成!")
        
        # 如果有matplotlib，显示预览
        try:
            preview_augmentations(data_root, num_samples=4, augment_strength='medium')
        except ImportError:
            print("⚠️ matplotlib未安装，跳过可视化预览")
    
    else:
        print(f"❌ 数据目录 {data_root} 不存在")