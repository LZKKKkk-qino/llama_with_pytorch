from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Union
from sentencepiece import SentencePieceProcessor
import time
import json

from torch.cuda import device


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    num_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # 这是FFN中隐藏层神经元数量的倍数约束，FFN隐藏层维度必须是该值的倍数（用于硬件计算对齐）
    ffn_dim_multiplier: Optional[float] = None

    # 构建 rope 位置编码矩阵和 kv_cache 使用
    max_batch_size: int = 32
    max_seq_len: int = 512

    norm_eps: float = 1e-5
    device: str = None

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 这里 self 的参数名称要和加载的 LLama 模型中每一层的参数名称一致
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.head_dim =args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = SwiGLUFeedForward(args)

        # Normalization
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # 先经过 attention， 再经过feed forward 层后得到当前 Block 的输出
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        output = h + self.feed_forward.forward(self.ffn_norm(h))
        return output


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.num_kv_heads is None else args.num_kv_heads
        self.num_heads_q = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 计算每个 query 对应多少个 kv head
        self.n_required = self.num_heads_q // self.n_kv_heads

        # q,k,v 权重
        self.wq = nn.Linear(args.dim, self.head_dim * args.n_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(self.head_dim * args.n_heads, args.dim, bias=False)

        # kv cache
        self.k_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.v_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))


    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        :param x: (batch_size, 1, dim), 每个batch输入的是一个token
        :param start_pos: 当前输入token的位置索引
        :param freqs_complex: ROPE位置编码矩阵
        :return: 加入位置编码attention的token embedding, shape = (batch_size, 1, dim)
        """
        # x 输入的是 1个 token , x shape (batch_size, 1, dim)
        # seq_len = 1
        batch_size, seq_len, dim = x.shape

        # q,k,v 矩阵
        x_q = self.wq(x).view(batch_size, seq_len, self.num_heads_q, self.head_dim)
        x_k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        x_v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用 ROPE 位置编码
        x_q = rotary_embedding(x_q, freqs_complex, x.device)
        x_k = rotary_embedding(x_k, freqs_complex, x.device)

        # 存入当前输入 x 的 k, v
        # 只索引前两个有确切值的维度，后面的维度自动
        self.k_cache[:batch_size, start_pos:start_pos + seq_len] = x_k
        self.v_cache[:batch_size, start_pos:start_pos + seq_len] = x_v

        # 从KV Cache 中获取全部的 K, V
        keys = self.k_cache[:batch_size, 0:start_pos + seq_len]
        values = self.v_cache[:batch_size, 0:start_pos + seq_len]

        # 计算 self-attention

        # shape (batch_size, seq_len = 1, n_heads, head_dim) -> (batch_size, n_heads, seq_len = 1, head_dim)
        x_q = x_q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 实现 MQA/GQA, 但 MQA 中通常仅使用 n_kv_heads=1，可通过broadcasting广播机制直接复制计算
        # 而 GQA 中则使用多个 n_kv_heads，需要根据倍数关系复制多个kv_head来计算
        if self.n_kv_heads != self.num_heads_q:
            # 重复每个 KV head n_required 次
            keys = torch.repeat_interleave(keys, repeats=self.n_required, dim=1)
            values = torch.repeat_interleave(values, repeats=self.n_required, dim=1)

        # attention scores
        # x_q @ keys
        # (batch_size, n_heads, 1, head_dim) @ (batch_size, n_kv_heads * n_required, head_dim, kv_seq_len)
        # = (batch_size, n_heads, 1, kv_seq_len)
        scores = torch.matmul(x_q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(x_q) # 在 xq 对 k_cache 每个词的 scores 维度上做 softmax

        # score @ values
        # (batch_size, n_heads, 1, kv_seq_len) @ (batch_size, n_heads, kv_seq_len, head_dim)
        # = (batch_size, n_heads, 1, head_dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.wo(output)

        return output  # shape = (batch_size, 1, dim)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = int(args.dim * 4 * 2 / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.dim * args.ffn_dim_multiplier)

        # 确保 hidden_dim 是 args.multiple_of 的倍数， 如果不是，则将 hidden_dim 向上取整一个 multiple_of 的倍数
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # SwiGLU 是 Swish 和 GELU 的组合
        # 当 Swish 中的 β = 1 时，SwiGLU 等价于 Swish
        # SwiGLU(a, b) = a * Swish(b)
        swish_x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(swish_x)
        return x


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str):
    # 计算论文中的(15)的 rotary matrix
    # 预先计算 ROPE 的 m 和 theta
    assert head_dim % 2 == 0  # 保证 head_dim 是偶数, ROPE 是每两个维度进行旋转位置编码的

    # m 表示每个token的位置索引
    # shape (seq_len)
    m = torch.arange(seq_len, device=device)

    # theta 表示每个token的旋转角度
    # shape (d/2)
    theta_numerate = torch.arange(0, head_dim, 2, device=device) # 2(i-1)
    theta = 1/ (10000 ** (theta_numerate / head_dim))  # 10000^(-2(i-1)/d)

    # 计算 m 和 theta 的内积，得到每个维度的旋转角度
    # shape (seq_len, d/2)
    freqs = torch.outer(m, theta).float()
    # 得到每个维度的旋转矩阵
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freq_complex

#
# f = precompute_theta_pos_frequencies(head_dim=64, seq_len=2048, device="cuda")
# print(f.shape)
# print(f[1,:3])
# x_real = torch.randn(2, 3, 4, 2)  # 实数张量,最后一维是2
# x_complex = torch.view_as_complex(x_real)
# print(x_complex.shape)  # 输出: torch.Size([2, 3, 4]),不是 [2, 3, 4, 1]


def rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: Union[str, torch.device]):
    """
    :param device: str or torch.device
    :param x: (batch_size, seq_len, dim)
    :param freqs_complex: (seq_len, dim/2)
    :return: (batch_size, seq_len, dim)
    """
    # 计算 x 每两个维度向量的复数表示
    # x_complex shape (batch_size, seq_len, dim/2)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)  # x.shape[:-1] 是 x shape 去掉最后一维的其他所有维度，是一个元组
        # *xq.shape[:-1] 前面的星号 * 是 Python 的“解包（Unpacking）”操作。
        # 它会把元组(a, b, c)拆成三个独立的数字：a, b, c。
        # 比如 (batch_size, seq_len, dim) 经过[:-1]后, 得到 (batch_size, seq_len)
    )  # reshape 操作输入就相当于是 (x.shape[:-1] = (batch_size, seq_len), -1, 2)

    # 计算 x 的旋转后的复数表示
    # (seq_len, dim/2) -> (1, seq_len, dim/2)
    freqs_complex = freqs_complex.unsqueeze(0)
    x_rotary_complex = x_complex * freqs_complex
    # 将计算出来的复数 a+bi 提取a, b实数形式，实部为 x1 旋转后的结果，虚部为 x2 旋转后的结果
    x_rotated = torch.view_as_real(x_rotary_complex)
    # 将 x_rotated 的维度从 (batch_size, seq_len, dim/2, 2) 重新恢复成 (batch_size, seq_len, dim)
    return x_rotated.reshape(*x.shape).type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 公式中的 g 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # rmsnorm也是layer norm, 让不同位置（seq)的归一化独立
        # return x * torch.rsqrt(x.pow(2).mean((0,1), keepdim=True) + self.eps) # batchnorm的话就是在batch和seq_len 维度上做归一化，让每个batch所有位置共享统计量

    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_layers = args.n_layers


        # embedding
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.args.dim)

        # Normalization
        self.norm = RMSNorm(self.args.dim, eps=args.norm_eps)

        # Layers

        # Encoder Layer
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(self.num_layers)])

        # Output Layer
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # 计算得到 ROPE 位置编码需要的 m 和 theta
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)

    def forward(self, token: torch.Tensor, start_pos: int):
        """
        LLama模型的前向传播过程
        :param tokens: (batch_size, seq_len)
        :param start_pos: 当前生成的起始位置（推理时使用）
        :return: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token.shape
        assert seq_len == 1  # 确保输入的是一个token

        x = self.tok_embeddings(token)
        # x = self.norm(x)

        # 根据token的位置[start_pos, start_pos+seq_len] 来获取 positional encoding 对应的 m 和 theta
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]
        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)
        x = self.norm(x)
        output = self.output(x).float()
        return output


