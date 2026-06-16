# LLaMA with PyTorch

一个基于 PyTorch 的 LLaMA (Large Language Model Meta AI) 模型实现，用于学习大语言模型的架构和原理。

## 📚 项目简介

本项目实现了论文 [《LLaMA: Open and Efficient Foundation Language Models》](https://arxiv.org/abs/2302.13971) 中描述的核心组件，包括：

- **RMSNorm** (Root Mean Square Layer Normalization)
- **RoPE** (Rotary Position Embeddings)
- **SwiGLU** 激活函数
- **Multi-Query Attention** (MQA) 和 **Grouped-Query Attention** (GQA)

这是一个学习项目，旨在通过实际编码深入理解 Transformer 架构和大语言模型的工作原理。

## 🎯 特性

- ✅ 完整的 Transformer 架构实现
- ✅ 支持 MQA/GQA 注意力机制
- ✅ RoPE 旋转位置编码
- ✅ Top-P 采样策略
- ✅ CPU/GPU 推理支持
- ✅ 批处理推理

## 📁 项目结构

```
llama_with_pytorch/
├── model.py       # 核心模型定义（Transformer Block、Attention、FeedForward 等）
├── inference.py   # 推理引擎（模型加载、生成逻辑）
├── main.py        # 使用示例
└── requirements.txt # Python 依赖
```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/llama_with_pytorch.git
cd llama_with_pytorch
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 💡 使用示例

```python
from inference import LLama

# 加载模型（需要提供模型权重路径）
model = LLama(
    model_path="path/to/model/weights",
    tokenizer_path="path/to/tokenizer.model",
    max_seq_len=2048,
    max_batch_size=4
)

# 生成文本
tokens = [1, 2, 3, 4]  # 输入 token 序列
output = model.generate(tokens, max_new_tokens=100)
```

## 🔧 核心组件说明

### ModelArgs
模型配置参数，包括维度、层数、注意力头数等。

### SelfAttention
自注意力机制实现，支持 Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA)。

### SwiGLUFeedForward
使用 SwiGLU 激活函数的前馈网络，比传统 ReLU 性能更好。

### RMSNorm
均方根归一化，比 Layer Normalization 更简洁高效。

### TransformerBlock
完整的 Transformer 编码器块，包含自注意力和前馈网络。

## 📖 学习资源

- [LLaMA 论文](https://arxiv.org/abs/2302.13971)
- [RoPE 位置编码论文](https://arxiv.org/abs/2104.09864)
- [PyTorch 官方文档](https://pytorch.org/docs/)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## ⚠️ 注意事项

- 本项目仅用于学习目的
- 需要自行下载 LLaMA 模型权重
- 推理需要足够的内存（取决于模型大小）