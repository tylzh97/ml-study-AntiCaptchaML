# CRNN验证码识别

基于PyTorch的CRNN（CNN+LSTM+CTC）验证码识别模型，支持4-8位长度不定的英文数字验证码识别。

## 特性

- 支持变长验证码识别（4-8位）
- 支持大小写英文字母和数字（62个字符）
- 使用CTC损失函数，无需字符级对齐
- 自动生成训练数据
- 完整的训练、推理和评估流程

## 项目结构

```
captcha_recognition/
├── data/                   # 数据目录
├── models/                 # 模型定义
│   ├── __init__.py
│   └── crnn.py            # CRNN模型
├── utils/                  # 工具函数
│   ├── __init__.py
│   └── dataset.py         # 数据集类
├── checkpoints/           # 模型权重
├── output/               # 输出结果
├── generate_data.py      # 数据生成脚本
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖包
└── README.md           # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成训练数据

```bash
python generate_data.py --output_dir ./data --num_samples 50000
```

参数说明：
- `--output_dir`: 数据保存目录
- `--num_samples`: 总样本数
- `--width`: 图片宽度（默认160）
- `--height`: 图片高度（默认60）

### 2. 训练模型

```bash
python train.py --data_root ./data --batch_size 32 --num_epochs 100 --lr 1e-3
```

主要参数：
- `--data_root`: 数据根目录
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--lr`: 学习率
- `--hidden_size`: LSTM隐藏层大小（默认256）
- `--num_layers`: LSTM层数（默认2）
- `--save_dir`: 模型保存目录
- `--resume`: 恢复训练的检查点路径

### 3. 推理预测

#### 单张图片预测
```bash
python inference.py --model_path ./checkpoints/best.pth --mode single --image_path test.png --visualize
```

#### 批量预测
```bash
python inference.py --model_path ./checkpoints/best.pth --mode batch --image_dir ./test_images --output_path results.txt
```

#### 模型评估
```bash
python inference.py --model_path ./checkpoints/best.pth --mode evaluate --data_root ./data --output_path eval_results.txt
```

## 模型架构

### CRNN结构
- **CNN特征提取**：7层卷积网络，输出512维特征
- **RNN序列建模**：2层双向LSTM，隐藏层256维
- **CTC解码**：处理变长序列识别

### 字符集
- 数字：0-9 (10个)
- 大写字母：A-Z (26个)  
- 小写字母：a-z (26个)
- 总计：62个字符

### 训练策略
- 损失函数：CTC Loss
- 优化器：Adam (lr=1e-3, weight_decay=1e-4)
- 学习率调度：StepLR (每20轮衰减0.5)
- 数据增强：颜色抖动
- 梯度裁剪：max_norm=5.0

## 性能指标

模型评估包含两个指标：
- **序列准确率**：整个验证码完全正确的比例
- **字符准确率**：单个字符正确的比例

## 示例结果

训练良好的模型通常可以达到：
- 序列准确率：85-95%
- 字符准确率：95-98%

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 数据生成时可以调整干扰程度
3. 训练时监控过拟合，适时调整学习率
4. CTC解码对序列长度敏感，注意输入图像尺寸

## 扩展建议

1. **数据增强**：添加更多噪声、模糊、旋转等
2. **模型改进**：尝试Attention机制、Transformer
3. **后处理**：添加语言模型纠错
4. **部署优化**：模型量化、ONNX转换