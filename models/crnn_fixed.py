import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedCRNN(nn.Module):
    def __init__(self, img_height, img_width, num_classes, hidden_size=256, num_layers=2):
        """
        修复的CRNN模型：优化序列长度匹配
        
        Args:
            img_height: 输入图像高度
            img_width: 输入图像宽度  
            num_classes: 字符类别数（不包括blank）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
        """
        super(FixedCRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # 优化的CNN架构 - 减少序列长度
        self.cnn = nn.Sequential(
            # Conv Block 1 - 下采样更激进
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 160x60 -> 80x30
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 80x30 -> 40x15
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 40x15 -> 20x7 (关键修改)
            
            # Conv Block 4 - 只在高度方向池化
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 20x7 -> 20x3
            
            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), (3, 1)),  # 20x3 -> 20x1 (强制高度为1)
        )
        
        # 计算CNN输出的特征图尺寸
        self._initialize_cnn_output_size()
        
        # RNN部分
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_height * 512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for blank
        
    def _initialize_cnn_output_size(self):
        """计算CNN输出的特征图尺寸"""
        dummy_input = torch.randn(1, 3, self.img_height, self.img_width)
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_height = cnn_output.size(2)
            self.cnn_output_width = cnn_output.size(3)
        print(f"修复模型 - CNN输出特征图尺寸: {cnn_output.shape}")
        print(f"序列长度: {self.cnn_output_width} (更适合4-8字符验证码)")
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, 3, height, width)
            
        Returns:
            log_probs: (seq_len, batch_size, num_classes+1)
        """
        batch_size = x.size(0)
        
        # CNN特征提取
        conv_features = self.cnn(x)  # (batch, 512, height, width)
        
        # 重塑为序列数据
        conv_features = conv_features.permute(0, 3, 2, 1)  # (batch, width, height, channels)
        seq_len = conv_features.size(1)
        conv_features = conv_features.contiguous().view(
            batch_size, seq_len, -1
        )  # (batch, width, height*channels)
        
        # RNN序列建模
        rnn_output, _ = self.rnn(conv_features)  # (batch, seq_len, hidden_size*2)
        
        # 分类
        output = self.classifier(rnn_output)  # (batch, seq_len, num_classes+1)
        
        # CTC需要的格式：(seq_len, batch, num_classes+1)
        output = output.permute(1, 0, 2)
        log_probs = F.log_softmax(output, dim=2)
        
        return log_probs


class CRNNLoss(nn.Module):
    def __init__(self):
        super(CRNNLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        CTC损失计算
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


if __name__ == "__main__":
    # 测试修复的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 62
    model = FixedCRNN(img_height=60, img_width=160, num_classes=num_classes)
    model.to(device)
    
    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 60, 160).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\n测试结果:")
        print(f"模型输出形状: {output.shape}")
        print(f"序列长度: {output.shape[0]} (预期10-25适合4-8字符)")
        print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 测试CTC兼容性
        seq_len, batch = output.shape[0], output.shape[1]
        input_lengths = torch.full((batch,), seq_len, dtype=torch.long)
        target_lengths = torch.tensor([4, 5, 6, 8], dtype=torch.long)  # 模拟目标长度
        
        print(f"CTC兼容性检查:")
        print(f"  输入长度: {input_lengths.tolist()}")
        print(f"  目标长度: {target_lengths.tolist()}")
        print(f"  长度比例: {seq_len/target_lengths.float().mean():.1f}:1 (理想范围2-4:1)")