# CRNN验证码识别

基于PyTorch的CRNN（CNN+LSTM+CTC）验证码识别模型，支持4-8位长度不定的英文数字验证码识别。

## 特性

- 🎯 支持变长验证码识别（4-8位）
- 🔤 支持大小写英文字母和数字（62个字符）
- 🧠 **双架构支持**：基础CNN + ResNet50预训练模型
- 🔧 **灵活训练策略**：支持骨干网络冻结和差分学习率
- 📊 使用CTC损失函数，无需字符级对齐
- 🤖 自动生成训练数据
- 🚀 完整的训练、推理和评估流程

## 项目结构

```
AntiCaptchaML/
├── data/                   # 数据目录（被.gitignore忽略）
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── crnn.py            # 基础CRNN模型
│   └── crnn_resnet.py     # ResNet-CRNN模型
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── dataset.py         # 数据集类
│   └── metrics.py         # 评估指标
├── checkpoints/           # 模型权重（被.gitignore忽略）
├── output/               # 输出结果（被.gitignore忽略）
├── generate_data.py      # 数据生成脚本
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── quick_start.py       # 快速开始脚本
├── requirements.txt     # 依赖包
└── README.md           # 说明文档
```

## 安装依赖

### 使用uv（推荐）
```bash
uv venv -p 3.12
uv pip install -r requirements.txt
```

### 使用pip
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速开始

```bash
# 一键完成数据生成、训练和评估
uv run python quick_start.py --num_samples 10000 --num_epochs 20

# 或者单独运行各个步骤...
```

### 2. 生成训练数据

```bash
# 生成5万个样本
uv run python generate_data.py --output_dir ./data --num_samples 50000

# 自定义图像尺寸
uv run python generate_data.py --output_dir ./data --num_samples 10000 --width 160 --height 60
```

**参数说明：**
- `--output_dir`: 数据保存目录
- `--num_samples`: 总样本数
- `--width`: 图片宽度（默认160）
- `--height`: 图片高度（默认60）

### 3. 模型训练

#### 🧠 ResNet50架构（推荐）

```bash
# 基础训练 - 冻结早期层
uv run python train.py --backbone resnet --freeze_backbone early --lr 1e-3 --batch_size 32 --num_epochs 50

# 使用SafeTensors格式保存模型
uv run python train.py --backbone resnet --freeze_backbone early --save_format safetensors

# 适用于大数据集 - 完全微调
uv run python train.py --backbone resnet --freeze_backbone none --lr 5e-4 --backbone_lr_ratio 0.5 --batch_size 16

# 适用于小数据集 - 冻结所有层
uv run python train.py --backbone resnet --freeze_backbone all --lr 1e-3 --batch_size 64

# 分阶段训练 - 只训练最后stage
uv run python train.py --backbone resnet --freeze_backbone partial --lr 1e-3 --backbone_lr_ratio 0.1
```

#### 📊 基础CNN架构

```bash
# 基础CRNN模型
uv run python train.py --backbone basic --lr 1e-3 --batch_size 32 --num_epochs 100
```

**核心参数说明：**
- `--backbone`: 模型架构 (`basic` | `resnet`)
- `--freeze_backbone`: ResNet冻结策略 (`none` | `all` | `early` | `partial`)
- `--backbone_lr_ratio`: ResNet相对学习率比例（默认0.1）
- `--pretrained`: 使用预训练权重（默认True）
- `--save_format`: 模型保存格式 (`pth` | `safetensors`)
- `--lr`: 基础学习率
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--hidden_size`: LSTM隐藏层大小（默认256）
- `--num_layers`: LSTM层数（默认2）

#### 🎯 不同数据量的推荐配置

```bash
# 小数据集 (<10k样本) - 避免过拟合
uv run python train.py --backbone resnet --freeze_backbone all --lr 1e-3 --batch_size 64

# 中等数据集 (10k-50k样本) - 平衡性能
uv run python train.py --backbone resnet --freeze_backbone early --lr 1e-3 --backbone_lr_ratio 0.1

# 大数据集 (>50k样本) - 充分利用数据
uv run python train.py --backbone resnet --freeze_backbone none --lr 5e-4 --backbone_lr_ratio 0.5
```

### 4. 推理预测

#### 单张图片预测
```bash
# 预测单张图片并可视化
uv run python inference.py --model_path ./checkpoints/best.pth --mode single --image_path test.png --visualize

# 简单预测
uv run python inference.py --model_path ./checkpoints/best.pth --mode single --image_path test.png
```

#### 批量预测
```bash
# 批量处理图片目录
uv run python inference.py --model_path ./checkpoints/best.pth --mode batch --image_dir ./test_images --output_path results.txt
```

#### 模型评估
```bash
# 在测试集上评估模型性能
uv run python inference.py --model_path ./checkpoints/best.pth --mode evaluate --data_root ./data --output_path eval_results.txt

# 使用SafeTensor格式模型进行推理
uv run python inference.py --model_path ./checkpoints/best.safetensors --mode single --image_path test.png
```

### 5. 模型格式转换

#### SafeTensors格式优势
- 🔒 **安全性**: 防止恶意代码注入，更安全的模型分发
- ⚡ **加载速度**: 比PyTorch格式更快的加载速度
- 💾 **存储优化**: 更紧凑的文件格式
- 🔄 **跨平台兼容**: 支持多种深度学习框架

#### 格式转换工具
```bash
# 单个模型转换
uv run python convert_to_safetensors.py ./checkpoints/best.pth

# 批量转换整个目录
uv run python convert_to_safetensors.py ./checkpoints/ --batch --output_dir ./safetensor_models
```

## 模型架构

### 🏗️ 双架构支持

#### ResNet-CRNN（推荐）
- **CNN骨干**：ResNet50预训练网络，提取高质量特征
- **自适应池化**：将特征图高度压缩为1，保持宽度作为序列
- **特征适配**：2048→512维度映射
- **RNN序列建模**：2层双向LSTM，隐藏层256维
- **CTC解码**：处理变长序列识别

#### 基础CRNN
- **CNN特征提取**：7层卷积网络，输出512维特征
- **RNN序列建模**：2层双向LSTM，隐藏层256维  
- **CTC解码**：处理变长序列识别

### 🎛️ 训练策略

#### 冻结策略对比
| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| `none` | 大数据集(>50k) | 性能最优 | 训练慢，易过拟合 |
| `early` | 中等数据集(10k-50k) | ⭐**推荐平衡选择** | 需要调参 |
| `partial` | 小-中数据集(5k-20k) | 稳定训练 | 特征受限 |
| `all` | 小数据集(<10k) | 快速收敛 | 性能受限 |

### 字符集
- 数字：0-9 (10个)
- 大写字母：A-Z (26个)  
- 小写字母：a-z (26个)
- 总计：62个字符

### 🔧 训练细节
- **损失函数**：CTC Loss（自动处理变长序列对齐）
- **优化器**：Adam优化器，支持差分学习率
- **学习率调度**：StepLR（每20轮衰减0.5）
- **数据增强**：颜色抖动（亮度、对比度、饱和度、色相）
- **梯度裁剪**：最大梯度范数5.0，防止梯度爆炸
- **正则化**：权重衰减1e-4，Dropout（LSTM层间）

## 性能指标

模型评估包含两个指标：
- **序列准确率**：整个验证码完全正确的比例
- **字符准确率**：单个字符正确的比例

## 📈 性能表现

### 预期结果
| 架构 | 数据量 | 序列准确率 | 字符准确率 | 训练时间 |
|------|--------|------------|------------|----------|
| 基础CRNN | 10k-50k | 80-88% | 92-95% | 2-4小时 |
| ResNet-CRNN | 10k-50k | **88-95%** | **95-98%** | 1-3小时 |
| ResNet-CRNN | >50k | **92-97%** | **97-99%** | 3-6小时 |

*以RTX 3090为基准，实际性能可能因数据质量和超参数调整而异*

## ⚠️ 使用须知

### 训练建议
1. **GPU内存**：ResNet模型需要4GB+显存，建议使用更大batch_size
2. **数据质量**：确保生成的验证码清晰度适中，过于复杂会影响收敛
3. **过拟合监控**：使用验证集监控，适时调整学习率和冻结策略
4. **序列长度**：CTC对输入序列长度敏感，保持图像宽高比一致

### 调参技巧
- 小数据集优先考虑冻结ResNet以防过拟合
- 大数据集可以降低学习率、增加训练轮数获得更好性能
- 差分学习率让ResNet用较小学习率微调，其他层用正常学习率

## 🚀 扩展方向

### 模型改进
- [ ] **Vision Transformer**：尝试ViT+Transformer解码器
- [ ] **Attention机制**：在LSTM后添加注意力层
- [ ] **多尺度特征**：融合不同分辨率的特征图

### 数据和后处理  
- [ ] **高级数据增强**：添加透视变换、弹性形变等
- [ ] **语言模型后处理**：使用字典和n-gram模型纠错  
- [ ] **半监督学习**：利用未标注的验证码图片

### 部署优化
- [ ] **模型量化**：INT8量化减少模型大小
- [ ] **ONNX导出**：跨平台推理部署
- [ ] **TensorRT优化**：GPU推理加速

---

## 📄 许可证

本项目仅供学习和研究使用。请勿用于破解他人验证码等恶意目的。