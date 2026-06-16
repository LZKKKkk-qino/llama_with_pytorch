# LLaMA with PyTorch - Claude 使用说明

## 项目简介

这是一个基于 PyTorch 的 LLaMA (Large Language Model Meta AI) 模型实现的学习项目。

项目实现了 LLaMA 论文中的核心组件：
- RMSNorm（均方根归一化）
- RoPE（旋转位置编码）
- SwiGLU 激活函数
- Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA)

## 📋 Claude 交互规则

### 语言设置
- **所有回答必须使用中文**

### 工作目录
- 主目录：`C:\Users\LZQIQINONO\Desktop\llama_with_pytorch`

## 📁 项目结构

```
llama_with_pytorch/
├── model.py       # 核心模型定义
├── inference.py   # 推理引擎
├── main.py        # 使用示例
├── requirements.txt # Python 依赖
├── README.md      # 项目文档
└── .gitignore     # Git 忽略配置
```

## 🔑 核心代码说明

### model.py
- `ModelArgs`: 模型配置参数
- `TransformerBlock`: Transformer 编码器块
- `SelfAttention`: 自注意力机制（支持 MQA/GQA）
- `SwiGLUFeedForward`: SwiGLU 前馈网络
- `RMSNorm`: 均方根归一化

### inference.py
- `LLama` 类：模型加载和推理
- 支持批处理
- Top-P 采样策略
- CPU/GPU 推理支持

## 📚 学习资源

- [LLaMA 论文](https://arxiv.org/abs/2302.13971)
- [RoPE 位置编码论文](https://arxiv.org/abs/2104.09864)

## ⚠️ 注意事项

- 本项目仅用于学习目的
- 需要自行下载 LLaMA 模型权重
- 推理需要足够的内存（取决于模型大小）