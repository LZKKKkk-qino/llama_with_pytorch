"""
LLaMA 推理示例脚本

这是一个简单的示例，展示如何使用本项目加载 LLaMA 模型并进行文本生成。
请确保你已经下载了 LLaMA 模型权重文件。
"""

import argparse
from inference import LLama
import torch


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='LLaMA 推理脚本')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='模型 checkpoint 目录路径（必需）')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer.model 文件路径（如果为空，则使用 checkpoint_dir/tokenizer.model）')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='最大序列长度')
    parser.add_argument('--max_batch_size', type=int, default=4,
                        help='最大批处理大小')
    parser.add_argument('--max_gen_len', type=int, default=100,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='采样温度（越高越随机）')
    parser.add_argument('--top_p', type=float, default=0.7,
                        help='Top-P 采样阈值')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备（cuda/cpu），默认自动检测')
    parser.add_argument('--prompt', type=str, default='The meaning of life is',
                        help='输入提示词')

    args = parser.parse_args()

    # 参数验证
    if args.checkpoint_dir is None:
        print("错误：请指定 --checkpoint_dir 参数")
        print("例如: python main.py --checkpoint_dir /path/to/llama-model")
        return

    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {args.device}")

    # 设置 tokenizer 路径
    if args.tokenizer_path is None:
        args.tokenizer_path = f"{args.checkpoint_dir}/tokenizer.model"

    # 加载模型
    print("=" * 50)
    print("正在加载模型...")
    print("=" * 50)
    model = LLama.build(
        checkpoints_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        load_model=True,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        device=args.device
    )
    print("=" * 50)
    print("模型加载完成！")
    print("=" * 50)

    # 运行推理
    print(f"\n输入提示词: {args.prompt}")
    print("=" * 50)
    print("生成结果:")
    print("=" * 50)

    output_tokens, output_texts = model.text_completion(
        prompt=[args.prompt],
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"\n{output_texts[0]}")
    print("=" * 50)

    # 打印统计信息
    print(f"\n生成 token 数量: {len(output_tokens[0])}")


if __name__ == '__main__':
    main()