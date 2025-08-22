import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_classes, hidden_size=256, num_layers=2):
        """
        CRNN模型：CNN + RNN + CTC
        
        Args:
            img_height: 输入图像高度
            img_width: 输入图像宽度
            num_classes: 字符类别数（不包括blank）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
        """
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN特征提取部分
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /4
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # /8 height, /4 width
            
            # Conv Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Conv Block 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # /16 height, /4 width
            
            # Conv Block 7 - 适应性卷积，确保高度降为1
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 计算CNN输出的特征图尺寸
        self._initialize_cnn_output_size()
        
        # RNN部分
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_height * 512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
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
        print(f"CNN输出特征图尺寸: {cnn_output.shape}")
        
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
        # 将宽度作为序列长度，高度和通道合并作为特征维度
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
        
        Args:
            log_probs: (seq_len, batch, num_classes+1)
            targets: (batch * target_seq_len,) - 拼接的目标序列
            input_lengths: (batch,) - 每个样本的输入序列长度
            target_lengths: (batch,) - 每个样本的目标序列长度
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 字符集大小：10个数字 + 26个大写字母 + 26个小写字母 = 62
    num_classes = 62
    model = CRNN(img_height=60, img_width=160, num_classes=num_classes)
    model.to(device)
    
    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 60, 160).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"模型输出形状: {output.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")