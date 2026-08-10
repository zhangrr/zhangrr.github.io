"""
从零训练中文 GPT-2

前置：已运行 03.tokenizer.py，得到 tokenized_data/tokenizer.json
数据：train_data_cleaned_chunks/*.txt

用法示例：
  python 04.train_gpt2.py
  python 04.train_gpt2.py --max-steps 5000 --block-size 512 --batch-size 8
  python 04.train_gpt2.py --model-size tiny --max-files 4  # 快速试跑
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_TOKENIZER_DIR = "tokenized_data"
DEFAULT_DATA_DIR = "train_data_cleaned_chunks"
DEFAULT_OUTPUT_DIR = "gpt2-zh-checkpoints"
DEFAULT_HF_TOKENIZER_DIR = "my-gpt2-tokenizer"

# 预置模型规模（从零训练，参数量可控）
MODEL_PRESETS = {
    # ~15M params，适合快速验证流水线
    "tiny": dict(n_embd=256, n_layer=4, n_head=4, n_positions=512),
    # ~40M params
    "small": dict(n_embd=512, n_layer=6, n_head=8, n_positions=512),
    # ~124M params，接近 GPT-2 small
    "base": dict(n_embd=768, n_layer=12, n_head=12, n_positions=1024),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Chinese GPT-2 from scratch.")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--hf-tokenizer-dir", default=DEFAULT_HF_TOKENIZER_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_PRESETS.keys()),
        default="small",
        help="模型规模预设：tiny / small / base",
    )
    parser.add_argument("--block-size", type=int, default=None, help="上下文长度，默认跟随模型预设")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000, help="训练步数，<=0 表示按 epoch 训练")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=0, help="只用前 N 个数据分片，0 表示全部")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="强制使用 fp16")
    parser.add_argument("--no-bf16", action="store_true", help="禁用 bf16（默认在支持时启用）")
    parser.add_argument("--resume", default=None, help="从 checkpoint 完整恢复（含优化器/步数）")
    parser.add_argument(
        "--init-from",
        default=None,
        help="仅加载模型权重继续训（新优化器与学习率，适合加长训练）",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"],
        help="学习率调度，加长训练推荐 cosine",
    )
    parser.add_argument("--skip-sample", action="store_true", help="训练结束后不生成样例")
    return parser.parse_args()


def load_or_convert_tokenizer(tokenizer_dir: str, hf_tokenizer_dir: str) -> PreTrainedTokenizerFast:
    """加载 tokenizers 库训练的 BPE，并保存为 HuggingFace 格式。"""
    tokenizer_json = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(
            f"找不到分词器文件: {tokenizer_json}\n请先运行 03.tokenizer.py"
        )

    # 与 03.tokenizer.py 中 special_tokens 对齐
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    # 长文会在 group_texts 中切成 block，这里放宽限制避免无用告警
    tokenizer.model_max_length = int(1e9)

    out = Path(hf_tokenizer_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out)
    print(f"✓ 分词器已加载并保存到 {out}/")
    print(f"  词表大小: {len(tokenizer)}")
    print(
        f"  BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, "
        f"PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}"
    )
    return tokenizer


def build_model(tokenizer: PreTrainedTokenizerFast, model_size: str, block_size: int) -> GPT2LMHeadModel:
    preset = MODEL_PRESETS[model_size]
    n_positions = block_size

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=preset["n_embd"],
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 创建 GPT-2 ({model_size})")
    print(
        f"  layers={config.n_layer}, embd={config.n_embd}, heads={config.n_head}, "
        f"ctx={config.n_positions}"
    )
    print(f"  参数量: {n_params / 1e6:.1f}M")
    return model


def resolve_data_files(data_dir: str, max_files: int) -> list[str]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据目录: {data_dir}\n请先运行 02.clean_data.py")

    files = sorted(str(p) for p in path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"数据目录为空: {data_dir}")

    if max_files > 0:
        files = files[:max_files]
    print(f"✓ 数据文件: {len(files)} 个")
    return files


def prepare_datasets(
    data_files: list[str],
    tokenizer: PreTrainedTokenizerFast,
    block_size: int,
    eval_ratio: float,
    num_workers: int,
):
    """加载文本 → 分词 → 打包成固定长度 block。"""
    print("加载文本数据集...")
    raw = load_dataset("text", data_files={"train": data_files}, split="train")
    print(f"  原始行数: {len(raw)}")

    # 去掉空行
    raw = raw.filter(lambda x: bool(x["text"] and x["text"].strip()), num_proc=max(1, num_workers))
    print(f"  非空行数: {len(raw)}")

    if eval_ratio > 0 and len(raw) > 100:
        split = raw.train_test_split(test_size=eval_ratio, seed=42)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw, eval_raw = raw, None

    def tokenize_function(examples):
        # 每行作为独立文档，末尾加 eos，便于打包
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts, add_special_tokens=False)

    print("分词中...")
    train_tok = train_raw.map(
        tokenize_function,
        batched=True,
        num_proc=max(1, num_workers),
        remove_columns=train_raw.column_names,
        desc="Tokenize train",
    )
    eval_tok = None
    if eval_raw is not None:
        eval_tok = eval_raw.map(
            tokenize_function,
            batched=True,
            num_proc=max(1, num_workers),
            remove_columns=eval_raw.column_names,
            desc="Tokenize eval",
        )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"打包成 block_size={block_size} ...")
    train_ds = train_tok.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=max(1, num_workers),
        desc="Group train",
    )
    eval_ds = None
    if eval_tok is not None:
        eval_ds = eval_tok.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=max(1, num_workers),
            desc="Group eval",
        )

    print(f"  训练 blocks: {len(train_ds)}")
    if eval_ds is not None:
        print(f"  验证 blocks: {len(eval_ds)}")
    return train_ds, eval_ds


def pick_precision(args) -> tuple[bool, bool]:
    """返回 (fp16, bf16)。"""
    if args.fp16:
        return True, False
    if args.no_bf16:
        return torch.cuda.is_available(), False
    # RTX 30/40/50 系列优先 bf16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True
    if torch.cuda.is_available():
        return True, False
    return False, False


def generate_samples(model, tokenizer, device, prompts=None, max_new_tokens=80):
    if prompts is None:
        prompts = [
            "深度学习是",
            "中国的首都是",
            "自然语言处理",
            "今天天气",
        ]

    model.eval()
    print("\n" + "=" * 60)
    print("生成样例")
    print("=" * 60)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n[提示] {prompt}")
        print(f"[输出] {text}")


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("中文 GPT-2 从零训练")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {mem_gb:.1f} GB")

    tokenizer = load_or_convert_tokenizer(args.tokenizer_dir, args.hf_tokenizer_dir)

    preset = MODEL_PRESETS[args.model_size]
    block_size = args.block_size or preset["n_positions"]
    # 不超过模型预设的位置编码长度
    block_size = min(block_size, preset["n_positions"])

    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"找不到 --init-from 路径: {init_path}")
        print(f"✓ 从已有权重热启动: {init_path}")
        model = GPT2LMHeadModel.from_pretrained(init_path)
        # 上下文长度需与数据 block 一致；若旧模型更短则仍用其 n_positions 截断
        model_ctx = int(model.config.n_positions)
        if block_size > model_ctx:
            print(f"  警告: block_size={block_size} > 模型 n_positions={model_ctx}，改为 {model_ctx}")
            block_size = model_ctx
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  已加载参数量: {n_params / 1e6:.1f}M, ctx={model_ctx}")
    else:
        model = build_model(tokenizer, args.model_size, block_size)

    data_files = resolve_data_files(args.data_dir, args.max_files)
    train_ds, eval_ds = prepare_datasets(
        data_files,
        tokenizer,
        block_size=block_size,
        eval_ratio=args.eval_ratio,
        num_workers=args.num_workers,
    )

    fp16, bf16 = pick_precision(args)
    print(f"精度: fp16={fp16}, bf16={bf16}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_eval = eval_ds is not None and len(eval_ds) > 0
    # 评估间隔：按 save_steps，且不超过总步数
    eval_steps = args.save_steps

    # transformers 5.x: 已移除 overwrite_output_dir / save_safetensors 等参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=eval_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        logging_first_step=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if use_eval else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"  output_dir: {args.output_dir}")
    print(f"  batch_size: {args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  block_size: {block_size}")
    print(f"  max_steps: {args.max_steps if args.max_steps > 0 else 'by epochs'}")
    print(f"  lr: {args.learning_rate} ({args.lr_scheduler})")
    if args.init_from:
        print(f"  init_from: {args.init_from}")
    if args.resume:
        print(f"  resume: {args.resume}")

    train_result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if use_eval:
        print("\n最终验证...")
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            try:
                eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
            except OverflowError:
                eval_metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print(f"\n✓ 模型已保存到 {args.output_dir}/")

    if not args.skip_sample:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        generate_samples(model, tokenizer, device)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"加载模型: from_pretrained('{args.output_dir}')")
    print("生成文本: python 05.generate.py")
    print("=" * 60)


if __name__ == "__main__":
    # 限制 tokenizer 并行警告
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
