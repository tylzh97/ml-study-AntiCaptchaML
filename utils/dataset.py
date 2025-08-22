import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import string


class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        验证码数据集
        
        Args:
            data_dir: 数据目录，应包含labels.txt文件
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 字符集：数字+大写字母+小写字母
        self.characters = string.digits + string.ascii_letters
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}  # +1 因为0是blank
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.characters)}
        self.idx_to_char[0] = '<blank>'  # blank标签
        
        self.num_classes = len(self.characters)  # 不包括blank
        print(f"字符集: {self.characters}")
        print(f"字符集大小: {self.num_classes}")
        
        # 读取标签文件
        labels_file = os.path.join(data_dir, 'labels.txt')
        self.samples = []
        
        with open(labels_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    image_name, text = line.split('\t')
                    image_path = os.path.join(data_dir, image_name)
                    if os.path.exists(image_path):
                        self.samples.append((image_path, text))
        
        print(f"加载样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 将文本转换为索引序列
        text_indices = [self.char_to_idx[char] for char in text]
        text_length = len(text_indices)
        
        return image, torch.tensor(text_indices, dtype=torch.long), text_length, text
    
    def decode_prediction(self, indices):
        """将预测的索引序列解码为文本"""
        chars = []
        prev_idx = 0
        
        for idx in indices:
            # CTC解码：跳过blank和重复字符
            if idx != 0 and idx != prev_idx:
                if idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])
            prev_idx = idx
        
        return ''.join(chars)


def collate_fn(batch):
    """
    自定义collate函数，处理变长序列
    """
    images, text_indices_list, text_lengths, texts = zip(*batch)
    
    # 图像可以直接stack
    images = torch.stack(images)
    
    # 拼接所有文本索引序列
    text_indices = torch.cat(text_indices_list)
    
    # 文本长度
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    
    return images, text_indices, text_lengths, texts


def create_data_transforms(img_height=60, img_width=160, is_train=True):
    """创建数据变换"""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(data_root, batch_size=32, num_workers=4, img_height=60, img_width=160):
    """创建数据加载器"""
    
    # 创建数据变换
    train_transform = create_data_transforms(img_height, img_width, is_train=True)
    val_transform = create_data_transforms(img_height, img_width, is_train=False)
    
    # 创建数据集
    train_dataset = CaptchaDataset(
        os.path.join(data_root, 'train'),
        transform=train_transform
    )
    
    val_dataset = CaptchaDataset(
        os.path.join(data_root, 'val'),
        transform=val_transform
    )
    
    test_dataset = CaptchaDataset(
        os.path.join(data_root, 'test'),
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


if __name__ == "__main__":
    # 测试数据集
    data_root = "./data"
    
    if os.path.exists(data_root):
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
            data_root, batch_size=4
        )
        
        print("测试数据加载器...")
        for batch_idx, (images, text_indices, text_lengths, texts) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Text indices shape: {text_indices.shape}")
            print(f"  Text lengths: {text_lengths}")
            print(f"  Texts: {texts}")
            
            if batch_idx == 0:
                break
    else:
        print(f"数据目录 {data_root} 不存在，请先生成数据")