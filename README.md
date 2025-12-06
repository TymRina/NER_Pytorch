# NER_Pytorch

基于PyTorch的中文文本命名实体识别（NER）系统。

## 项目概述

本项目实现了一个专为中文文本分析设计的完整NER系统。它提供了从数据预处理到模型训练、验证和推理的端到端功能，并配有用户友好的图形界面，便于交互使用。

## 功能特性

- **数据预处理**：高效处理中文NER任务的文本数据集
- **模型训练与验证**：完整的NER模型训练和评估流程
- **实体识别**：准确识别中文文本中的命名实体
- **长文本支持**：稳健处理长文档并正确跟踪实体位置
- **图形界面**：基于PyQt5的直观GUI，方便用户交互
- **预训练模型**：包含已训练的模型权重，可直接使用

## 技术栈

- **框架**：PyTorch
- **模型架构**：基于LSTM的神经网络
- **图形界面**：PyQt5
- **开发语言**：Python 3.7+

## 安装说明

### 前置要求

- Python 3.7 或更高版本
- PyTorch 1.6 或更高版本
- PyQt5

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/TymRina/NER_Pytorch.git
   cd NER_Pytorch
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

   *注意：如果requirements.txt不可用，请手动安装依赖：*
   ```bash
   pip install torch torchvision torchaudio
   pip install PyQt5
   ```

## 使用方法

### 1. 数据预处理

```bash
python 01_data_preprocess.py
```

该脚本将原始数据处理为模型训练所需的格式。

### 2. 模型训练

```bash
python 02_train.py
```

或使用提供的PowerShell脚本：
```powershell
./run_training.ps1
```

训练后的模型和参数将保存在`runs`目录中。

### 3. 模型验证

```bash
python 03_val.py
```

该脚本在验证集上评估训练模型的性能。

### 4. 实体识别推理

```bash
python 04_infer.py
```

该脚本提供命令行接口用于实体识别。

### 5. 图形界面

```bash
python 05_pyqt_interface.py
```

启动用户友好的GUI进行交互式实体识别。

## 项目结构

```
NER_Pytorch/
├── 01_data_preprocess.py    # 数据预处理脚本
├── 02_train.py              # 模型训练脚本
├── 03_val.py                # 模型验证脚本
├── 04_infer.py              # 推理脚本
├── 05_pyqt_interface.py     # PyQt图形界面
├── model.py                 # 神经网络模型定义
├── loss_related_code.py     # 损失函数实现
├── run_training.ps1         # 训练PowerShell脚本
├── dataset/                 # 处理后的数据集文件
├── NER命名实体识别数据集/     # 原始数据集文件
└── runs/                    # 模型权重和参数
    ├── best_model.pth       # 最佳训练模型
    ├── loss_curve.png       # 训练损失曲线
    ├── model_params.json    # 模型参数
    ├── tag_map.json         # 实体标签映射
    └── vocab.json           # 词汇词典
```

## 模型详情

NER模型基于双向LSTM架构，包含以下关键组件：

- 用于字符表示的嵌入层
- 用于上下文特征提取的双向LSTM层
- 带有CRF（条件随机场）的线性层，用于序列标注

## 性能表现

该模型在识别中文文本中各种类型的命名实体方面具有较高的准确性，包括：
- 人名
- 地点
- 组织
- 其他自定义实体类型

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 致谢

- 本项目基于PyTorch构建，PyTorch为深度学习研究和开发提供了出色的支持。
- CRF实现改编自最先进的NER研究成果。

## 贡献

欢迎贡献！请随时提交Pull Request。

## 联系方式

如有任何问题或疑问，请在GitHub上提交issue。
