import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetCRNN(nn.Module):
    def __init__(self, img_height, img_width, num_classes, hidden_size=256, num_layers=2, pretrained=True):
        """
        基于ResNet50的CRNN模型：ResNet50 + RNN + CTC
        
        Args:
            img_height: 输入图像高度
            img_width: 输入图像宽度
            num_classes: 字符类别数（不包括blank）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            pretrained: 是否使用预训练权重
        """
        super(ResNetCRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # 使用ResNet50作为CNN骨干网络
        if pretrained:
            # 尝试加载预训练权重，支持离线模式
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                print("✓ 成功加载在线预训练权重")
            except Exception as e:
                print(f"⚠️ 在线加载失败: {e}")
                # 尝试从本地加载
                local_model_path = "./pretrained/resnet50-0676ba61.pth"
                if os.path.exists(local_model_path):
                    resnet = models.resnet50(weights=None)
                    state_dict = torch.load(local_model_path, map_location='cpu')
                    resnet.load_state_dict(state_dict)
                    print(f"✓ 成功从本地加载预训练权重: {local_model_path}")
                else:
                    print("❌ 未找到本地预训练权重，使用随机初始化")
                    resnet = models.resnet50(weights=None)
        else:
            resnet = models.resnet50(weights=None)
        
        # 移除ResNet的全连接层和全局平均池化
        self.cnn = nn.Sequential(
            resnet.conv1,      # (3, H, W) -> (64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # (64, H/2, W/2) -> (64, H/4, W/4)
            resnet.layer1,     # (64, H/4, W/4) -> (256, H/4, W/4)
            resnet.layer2,     # (256, H/4, W/4) -> (512, H/8, W/8)
            resnet.layer3,     # (512, H/8, W/8) -> (1024, H/16, W/16)
            resnet.layer4,     # (1024, H/16, W/16) -> (2048, H/32, W/32)
        )
        
        # 添加自适应层来调整特征图尺寸
        # 目标：将高度压缩为1或2，保持宽度用于序列建模
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # 高度压缩为1，宽度保持
        
        # 特征维度适配层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
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
        
        # 初始化新添加的层
        self._initialize_weights()
    
    def freeze_backbone(self, freeze_mode='none'):
        """
        冻结ResNet骨干网络的不同部分
        
        Args:
            freeze_mode: 冻结模式
                - 'none': 不冻结任何层
                - 'all': 冻结所有ResNet层
                - 'early': 冻结早期层（conv1, layer1, layer2）
                - 'partial': 冻结前三个stage（conv1, layer1, layer2, layer3）
        """
        if freeze_mode == 'none':
            for param in self.cnn.parameters():
                param.requires_grad = True
            print("ResNet: 所有层可训练")
            
        elif freeze_mode == 'all':
            for param in self.cnn.parameters():
                param.requires_grad = False
            print("ResNet: 所有层冻结")
            
        elif freeze_mode == 'early':
            # 冻结前两个stage
            modules_to_freeze = ['0', '1', '2', '3', '4', '5']  # conv1, bn1, relu, maxpool, layer1, layer2
            for i, module in enumerate(self.cnn):
                if i in [0, 1, 2, 3, 4, 5]:  # conv1, bn1, relu, maxpool, layer1, layer2
                    for param in module.parameters():
                        param.requires_grad = False
            print("ResNet: 冻结早期层（conv1, layer1, layer2），layer3和layer4可训练")
            
        elif freeze_mode == 'partial':
            # 冻结前三个stage
            for i, module in enumerate(self.cnn):
                if i in [0, 1, 2, 3, 4, 5, 6]:  # conv1, bn1, relu, maxpool, layer1, layer2, layer3
                    for param in module.parameters():
                        param.requires_grad = False
            print("ResNet: 冻结前三个stage，只有layer4可训练")
        
        # 显示可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"参数统计: 总计{total_params:,}, 可训练{trainable_params:,}, 冻结{frozen_params:,}")
        print(f"可训练比例: {trainable_params/total_params*100:.1f}%")
    
    def get_param_groups(self, backbone_lr_ratio=0.1):
        """
        获取不同学习率的参数组
        
        Args:
            backbone_lr_ratio: 骨干网络相对于其他部分的学习率比例
        
        Returns:
            参数组列表，用于优化器
        """
        backbone_params = []
        other_params = []
        
        # 分离ResNet参数和其他参数
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('cnn.'):
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr_ratio': 1.0},
            {'params': backbone_params, 'lr_ratio': backbone_lr_ratio}
        ]
        
        print(f"参数组: 其他参数{len(other_params)}, ResNet参数{len(backbone_params)}")
        print(f"ResNet学习率比例: {backbone_lr_ratio}")
        
        return param_groups
        
    def _initialize_weights(self):
        """初始化新添加层的权重"""
        for m in [self.feature_adapter, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def _initialize_cnn_output_size(self):
        """计算CNN输出的特征图尺寸"""
        dummy_input = torch.randn(1, 3, self.img_height, self.img_width)
        with torch.no_grad():
            # ResNet特征提取
            resnet_features = self.cnn(dummy_input)
            # 自适应池化
            pooled_features = self.adaptive_pool(resnet_features)
            # 特征适配
            adapted_features = self.feature_adapter(pooled_features)
            
            self.cnn_output_height = adapted_features.size(2)
            self.cnn_output_width = adapted_features.size(3)
            
        print(f"ResNet特征图尺寸: {resnet_features.shape}")
        print(f"池化后特征图尺寸: {pooled_features.shape}")
        print(f"最终CNN输出特征图尺寸: {adapted_features.shape}")
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, 3, height, width)
            
        Returns:
            log_probs: (seq_len, batch_size, num_classes+1)
        """
        batch_size = x.size(0)
        
        # ResNet特征提取
        resnet_features = self.cnn(x)  # (batch, 2048, H/32, W/32)
        
        # 自适应池化 - 压缩高度维度
        pooled_features = self.adaptive_pool(resnet_features)  # (batch, 2048, 1, W')
        
        # 特征维度适配
        conv_features = self.feature_adapter(pooled_features)  # (batch, 512, 1, W')
        
        # 重塑为序列数据
        # 将宽度作为序列长度，高度和通道合并作为特征维度
        conv_features = conv_features.permute(0, 3, 2, 1)  # (batch, W', 1, 512)
        seq_len = conv_features.size(1)
        conv_features = conv_features.contiguous().view(
            batch_size, seq_len, -1
        )  # (batch, W', 512)
        
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


# 为了兼容性，保持原有的CRNN类
class CRNN(ResNetCRNN):
    """CRNN别名，指向ResNetCRNN以保持向后兼容"""
    def __init__(self, img_height, img_width, num_classes, hidden_size=256, num_layers=2, pretrained=True):
        super(CRNN, self).__init__(img_height, img_width, num_classes, hidden_size, num_layers, pretrained)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 字符集大小：10个数字 + 26个大写字母 + 26个小写字母 = 62
    num_classes = 62
    
    # 测试ResNet-CRNN模型
    print("\n测试ResNet-CRNN模型:")
    model = ResNetCRNN(img_height=60, img_width=160, num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # 测试输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 60, 160).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"模型输出形状: {output.shape}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
    # 检查ResNet预训练权重是否加载
    if hasattr(model.cnn[0], 'weight'):
        print(f"ResNet conv1权重统计: mean={model.cnn[0].weight.mean().item():.6f}, std={model.cnn[0].weight.std().item():.6f}")
    
    print("\n模型构建成功！")