"""
使用训练好的中文 GPT-2 生成文本

用法：
  python 05.generate.py
  python 05.generate.py --prompt "深度学习是" --max-new-tokens 100
  python 05.generate.py --model-dir gpt2-zh-checkpoints --temperature 0.9
"""
from __future__ import annotations

import argparse

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with trained Chinese GPT-2.")
    parser.add_argument("--model-dir", default="gpt2-zh-checkpoints")
    parser.add_argument("--prompt", default=None, help="单条提示；省略则使用内置多条提示")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return", type=int, default=1, help="每个提示生成几条")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {args.model_dir} (device={device})")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    prompts = (
        [args.prompt]
        if args.prompt
        else [
            "深度学习是",
            "中国的首都是",
            "自然语言处理",
            "今天天气",
            "人工智能技术",
        ]
    )

    print("=" * 60)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                num_return_sequences=args.num_return,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        print(f"\n[提示] {prompt}")
        for i, seq in enumerate(outputs):
            text = tokenizer.decode(seq, skip_special_tokens=True)
            prefix = f"  [{i+1}] " if args.num_return > 1 else "  "
            print(f"{prefix}{text}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
