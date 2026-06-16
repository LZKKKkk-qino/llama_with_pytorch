import json
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
from sentencepiece import SentencePieceProcessor

from model import TransformerBlock, ModelArgs


class LLama:
    def __init__(self, model: TransformerBlock, tokenizer: SentencePieceProcessor, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool , max_seq_len: int, max_batch_size: int, device: str):
        previous_time = time.time()

        # 加载模型 checkpoint
        if load_model:
            checkpoints_paths = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints_paths) > 0, f"No checkpoints found in {checkpoints_dir}"

            checkpoint = torch.load(checkpoints_paths[-1], map_location='cpu')
            print(f"加载 checkpoint 文件 Loaded checkpoint from {checkpoints_paths[-1]} in {time.time() - previous_time:.2f}s")

        # 设置 model 默认 device
        # 设置默认数据类型（只影响浮点数，不影响整数）
        if device == "cuda":
            # if torch.cuda.is_bf16_supported():
            #     torch.set_default_dtype(torch.bfloat16)
            # else:
            #     torch.set_default_dtype(torch.float16)
            # torch.set_default_dtype(torch.float16)
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            # CPU 使用 FP32 保证兼容性和数值稳定性
            # torch.set_default_dtype(torch.float32)
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        previous_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(max_seq_len=max_seq_len,
                               max_batch_size=max_batch_size,
                               device=device,
                               **params)
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size() # 覆盖原始 params 文件中的 vocab_size = -1，确保vocab_size 值正确载入
        model = TransformerBlock(model_args).to(device)

        if load_model:
            # 在我们的模型中已经计算了 rope 的 freqs, 所以这里不需要加载
            del checkpoint['rope.freqs']
            # if device == "cuda":
            #     torch.cuda.empty_cache()

            # strict = True 模型参数是字典,要键值 key 完全匹配
            model.load_state_dict(checkpoint, strict=True)
            print(f"加载模型参数 load state dict in {time.time() - previous_time:.2f}s")

        return LLama(model, tokenizer, model_args)

    def text_completion(self, prompt: list[str], max_gen_len: Optional[int] = None,
                        temperature: float = 0.9, top_p: float = 0.7, do_sample: bool = True):

        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len

        # 将 prompt 转为 token_ids
        prompt_ids = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompt]
        print(f"prompt_ids : {prompt_ids}")

        # batch_size 输入的 prompt 数量
        batch_size = len(prompt_ids)
        assert batch_size <= self.args.max_batch_size, f"Batch size: {batch_size} is larger than max batch size {self.args.max_batch_size}"

        max_prompt_len = max(len(i) for i in prompt_ids)
        print(f"max_prompt_len : {max_prompt_len}")
        assert max_prompt_len <= self.args.max_seq_len, f"Prompt length: {max_prompt_len} is larger than max seq length {self.args.max_seq_len}"

        # kv_cache 中 所能储存的最大长度为 max_seq_len, 所以 prompt + 模型生成的 tokens 长度不能超过 max_seq_len, 因此取二者最小值
        total_len = min(max_prompt_len + max_gen_len, self.args.max_seq_len)

        # 先用 pad_id 来填充 tokens_cache，后续用已有的和生成的 token来填充
        pad_id = self.tokenizer.pad_id()
        # if pad_id == -1:
        #     pad_id = self.tokenizer.bos_id()  # LLaMA 用 BOS 作为 padding
        tokens_cache = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)

        # 将 prompt 填充 tokens_cache
        for i, t_id in enumerate(prompt_ids):
            tokens_cache[i, :len(t_id)] = torch.tensor(t_id, dtype=torch.long, device=self.args.device)

        # 去判断 模型在每个batch中是否生成的 eos token，如果模型生成eos
        eos_reach = torch.tensor([False] * batch_size, device=self.args.device)

        # prompt 的 token 位置对应位置为 True
        prompt_tokens_mask = tokens_cache != pad_id
        print(f"prompt_tokens_mask : {prompt_tokens_mask.tolist()}")

        # 只需要取 total_len-1 个 tokens_cache, 因为 total_len 中的最后一个 token 是由前一个词预测出来的，得到最后一个词后不用继续生成了
        for pos in tqdm(range(1, total_len), desc="Generating"):
            with torch.no_grad():
                logits = self.model.forward(tokens_cache[:, pos-1:pos], pos-1)

                if do_sample:
                    # 基于 Top P 的随机采样策略
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    predict = self._sample_top_p(probs, top_p)
                else:
                    # 取当前这一个时刻最大的值对应索引, Greedy Search
                    predict = torch.argmax(logits[:, -1], dim=-1).reshape(-1)


            # 确保原本prompt中的词不会被模型预测的词覆盖，如果是推理超过了prompt的地方才将预测的词保存
            predict = torch.where(prompt_tokens_mask[:, pos], tokens_cache[:, pos], predict)
            tokens_cache[:, pos] = predict

            # 判断是否生成了eos
            # 右边等式:当有一个词生成了eos(Ture)且不是prompt的token (True)位置时，则 & 为 True
            # 左边等式:当模型回复输出生成了eos token，则 | 运算为 True, 代表该batch 生成了eos，其中全部都生成了eos则结束
            eos_reach |= (~prompt_tokens_mask[:, pos]) & (predict == self.tokenizer.eos_id())

            if all(eos_reach):
                print('token 生成结束')
                break
        print(f"tokens_cache : {tokens_cache.tolist()}")
        output_tokens = []
        output_texts = []

        # 按 batch 遍历，每次得到一个 batch 的完整 token 序列
        for batch_i, t_id in enumerate(tokens_cache.tolist()):
            if self.tokenizer.eos_id() in t_id:
                # eos_index = t_id.index(self.tokenizer.eos_id())
                # t_id = t_id[:eos_index]
                t_id = t_id[:t_id.index(self.tokenizer.eos_id())]
            output_tokens.append(t_id)
            output_texts.append(self.tokenizer.Decode(t_id))
        print(f"pad_token id ： {self.tokenizer.pad_id()}")
        print(f"eos_token id : {self.tokenizer.eos_id()}")
        return output_tokens, output_texts

    def _sample_top_p(self, probs: torch.Tensor, top_p: float):
        # probs_idx 是排序后概率值对应的原有的词典中的顺序
        # probs_sort 是排序后的概率值
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        # 没有被 top p 选择的 tokens 的概率值 全部设置为 0.0
        probs_sort[mask] = 0.0
        # 重新获得一个加起来是 1 的概率分布
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 下面进行随机采样
        # 从 top p distribution 中采样一个 token 的 index, 注意此时 index 并不是词典中对于的 token_id
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # 因为一开始概率分布 prob 进行了排序, 所以采样出来的 token index 并非词典中对应的 Token id
        # 这就是为什么一开始 sort 要接收两个东西的原因, probs_sort 是根据顺序排序的概率值,

        # 从 sorted 后的序列被取出的 index 来找到该 token 对应在词典中原来的 token id
        next_token = torch.gather(input=probs_idx, dim=-1, index=next_token)

        return next_token





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    checkpoint_dir = f"C:/Users/NONOQiQi/Downloads/llama-2-7b"
    model = LLama.build(checkpoints_dir=checkpoint_dir,
                        tokenizer_path=checkpoint_dir + "/tokenizer.model",
                        load_model=True,
                        max_seq_len=1024,
                        max_batch_size=2,
                        device=device)
    print('完成模型文件加载')

    # prompt = ['Please tell me how a logistics major graduate can find a job related to large language models.',
    #           'I want to become a ai model application engineer. My major is logistics, not computer science, but I want to engage in applications and development related to large models. How should I seek a job?']
    # prompt = [
    #     "Simply put, the theory of relativity states that ",
    #     "1+1=2 2+3=4 4+10= "
    # ]

    prompt = ['tell me the answer :1 plus 1 equal what',
              'tell me the answer :what is the capital of China?']
    # output_tokens, output_texts = (model.text_completion(prompt, max_gen_len=64, temperature=0.9, top_p=0.7))
    output_tokens, output_texts = model.text_completion(prompt, max_gen_len=64, temperature=0.9, top_p=0.7)
    print(output_tokens[0])
    assert len(output_texts) == len(prompt)
    for i in range(len(output_texts)):
        print(f"{output_texts[i]}")
        print("*"* 50)