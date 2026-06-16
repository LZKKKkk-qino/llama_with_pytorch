# LLaMA with PyTorch

一个基于 PyTorch 的 LLaMA (Large Language Model Meta AI) 模型实现，用于学习大语言模型的架构和原理。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Llama_with_PyTorch-green.svg)](https://github.com/LZKKKkk-qino/llama_with_pytorch)

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

- Python >= 3.10
- PyTorch >= 2.0.0

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/LZKKKkk-qino/llama_with_pytorch.git
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

### 运行方式

项目提供三种运行方式，适用于不同场景：

#### 方式一：命令行运行（推荐）

使用 `main.py` 通过命令行参数运行：

```bash
python main.py --checkpoint_dir /path/to/llama-model --prompt "请介绍一下人工智能"
```

**可选参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_dir` | 必需 | 模型 checkpoint 目录路径 |
| `--tokenizer_path` | 自动查找 | tokenizer.model 文件路径 |
| `--max_seq_len` | 1024 | 最大序列长度 |
| `--max_batch_size` | 4 | 最大批处理大小 |
| `--max_gen_len` | 100 | 最大生成长度 |
| `--temperature` | 0.9 | 采样温度（越高越随机） |
| `--top_p` | 0.7 | Top-P 采样阈值 |
| `--device` | 自动检测 | 运行设备（cuda/cpu） |
| `--prompt` | "The meaning of life is" | 输入提示词 |

#### 方式二：IDE 直接运行

在 PyCharm/VSCode 等 IDE 中直接运行 `inference.py`：

修改 `inference.py` 底部的配置后点击运行：

```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "你的模型路径"  # 修改这里

    model = LLama.build(
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=checkpoint_dir + "/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=2,
        device=device
    )

    prompt = ['你好，请介绍一下自己']  # 修改提示词
    output_tokens, output_texts = model.text_completion(
        prompt, max_gen_len=64, temperature=0.9, top_p=0.7
    )

    for i in range(len(output_texts)):
        print(f"{output_texts[i]}")
```

#### 方式三：自定义脚本导入使用

创建新的 Python 文件（如 `run.py`）并导入使用：

```python
from inference import LLama
import torch

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "你的模型路径"

# 加载模型
model = LLama.build(
    checkpoint_dir=checkpoint_dir,
    tokenizer_path=checkpoint_dir + "/tokenizer.model",
    load_model=True,
    max_seq_len=1024,
    max_batch_size=2,
    device=device
)

# 运行推理
prompt = ['解释什么是机器学习']
output_tokens, output_texts = model.text_completion(
    prompt, max_gen_len=100, temperature=0.7, top_p=0.9
)

print(output_texts[0])
```

### 推荐场景

| 场景 | 推荐方式 |
|------|----------|
| 快速测试 | 方式一（命令行） |
| 学习调试 | 方式二或方式三（IDE） |
| 批量处理 | 方式三（自定义脚本） |
| 生产部署 | 方式三（封装成服务） |

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