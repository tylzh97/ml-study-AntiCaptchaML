# 离线环境部署指南

## 📥 步骤1: 在有网络的机器上下载模型

### 方法1: 使用下载脚本（推荐）
```bash
# 自动下载并配置
uv run python download_pretrained.py
```

### 方法2: 手动下载
```bash
# 下载ResNet50预训练模型
wget https://download.pytorch.org/models/resnet50-0676ba61.pth

# 或使用浏览器下载
# 链接: https://download.pytorch.org/models/resnet50-0676ba61.pth
```

## 📦 步骤2: 准备离线部署包

### 创建目录结构
```bash
# 创建预训练模型目录
mkdir -p pretrained/

# 移动模型文件
mv resnet50-0676ba61.pth pretrained/
```

### 打包项目
```bash
# 打包整个项目（包含预训练模型）
tar -czf captcha-crnn-offline.tar.gz \
    models/ \
    utils/ \
    pretrained/ \
    *.py \
    requirements.txt \
    README.md \
    OFFLINE_DEPLOYMENT.md
```

## 🚀 步骤3: 在内网环境部署

### 解压和安装
```bash
# 解压项目
tar -xzf captcha-crnn-offline.tar.gz
cd captcha-crnn

# 创建虚拟环境（如果支持）
python3 -m venv venv
source venv/bin/activate

# 安装依赖（需要离线pip包或内网pypi镜像）
pip install -r requirements.txt
```

### 配置预训练模型

#### 选项1: 放置到PyTorch缓存目录
```bash
# 创建缓存目录
mkdir -p ~/.cache/torch/hub/checkpoints/

# 复制模型文件
cp pretrained/resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints/
```

#### 选项2: 使用项目本地路径（推荐）
模型已自动配置为优先从 `./pretrained/` 目录加载，无需额外配置。

## 🧪 步骤4: 测试部署

### 测试模型加载
```bash
# 测试ResNet模型创建
python -c "
from models.crnn_resnet import ResNetCRNN
model = ResNetCRNN(60, 160, 62, pretrained=True)
print('✅ ResNet-CRNN模型创建成功')
"
```

### 生成测试数据并训练
```bash
# 生成小量数据测试
python generate_data.py --num_samples 100 --output_dir ./test_data

# 测试训练
python train.py --data_root ./test_data --backbone resnet --num_epochs 2 --batch_size 4
```

## 🔧 配置选项

### 模型搜索路径优先级
1. **在线下载** (如果网络可用)
2. **项目本地路径**: `./pretrained/resnet50-0676ba61.pth`
3. **PyTorch缓存**: `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`
4. **随机初始化** (如果以上都失败)

### 训练命令示例
```bash
# 使用预训练ResNet (推荐)
python train.py --backbone resnet --pretrained --freeze_backbone early

# 不使用预训练权重
python train.py --backbone resnet --pretrained=False

# 使用基础CNN架构
python train.py --backbone basic
```

## ❗ 常见问题

### Q: 提示找不到预训练模型
**A:** 确保文件路径正确：
```bash
ls -la pretrained/resnet50-0676ba61.pth
# 应该显示文件存在且大小约93MB
```

### Q: 加载模型时出现权重不匹配错误
**A:** 确认下载的模型文件完整：
```bash
# 检查文件大小（应该约为93-98MB）
du -h pretrained/resnet50-0676ba61.pth
```

### Q: 内网环境无法安装PyTorch
**A:** 可以考虑：
1. 使用内网PyPI镜像源
2. 在外网机器上创建完整的离线包
3. 使用Docker容器方式部署

## 📊 验证部署成功

成功部署后，应该看到类似输出：
```
✓ 成功从本地加载预训练权重: ./pretrained/resnet50-0676ba61.pth
ResNet: 冻结早期层（conv1, layer1, layer2），layer3和layer4可训练
参数统计: 总计27,883,950, 可训练8,623,918, 冻结19,260,032
可训练比例: 30.9%
```

## 🎯 推荐配置

对于内网环境，推荐使用冻结策略减少训练复杂度：
```bash
# 小数据集 - 全冻结
python train.py --backbone resnet --freeze_backbone all --lr 1e-3

# 中等数据集 - 冻结早期层  
python train.py --backbone resnet --freeze_backbone early --lr 1e-3

# 大数据集 - 部分冻结
python train.py --backbone resnet --freeze_backbone partial --lr 1e-3
```