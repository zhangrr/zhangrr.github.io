"""
中文问答监督微调（SFT）

在预训练 GPT-2 上，用「问题→答案」数据做因果语言建模微调。
默认对问题部分 mask loss（labels=-100），只训练答案生成。

用法：
  python 06.sft_qa.py
  python 06.sft_qa.py --model-dir gpt2-zh-base-checkpoints --epochs 5
  python 06.sft_qa.py --prompt "中国的首都是"   # 仅测试生成
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL_DIR = "gpt2-zh-base-checkpoints"
DEFAULT_QA_FILE = "qa_data/zh_qa.jsonl"
DEFAULT_OUTPUT_DIR = "gpt2-zh-sft-qa"
DEFAULT_REPLAY_FILE = "qa_data/qa_replay.txt"


PROMPT_TEMPLATES = [
    "### 问题：\n{q}\n### 答案：\n",
    "问题：{q}\n答案：",
    "问：{q}\n答：",
    "Q: {q}\nA: ",
    "{q}",
]


def parse_args():
    p = argparse.ArgumentParser(description="SFT Chinese GPT-2 on Q&A data.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--qa-file", default=DEFAULT_QA_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--replay-file", default=DEFAULT_REPLAY_FILE, help="导出纯文本供续训回放")
    p.add_argument("--epochs", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0 时覆盖 epochs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--augment", type=int, default=3, help="每条样本用几种模板扩写")
    p.add_argument("--mask-prompt", action="store_true", default=True, help="只对答案算 loss")
    p.add_argument("--no-mask-prompt", action="store_true", help="整段都算 loss")
    p.add_argument("--skip-train", action="store_true", help="跳过训练只做生成")
    p.add_argument("--prompt", default=None, help="训练后/仅测试时的生成提示")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=40)
    return p.parse_args()


def load_qa_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    if not pairs:
        raise ValueError(f"问答数据为空: {path}")
    return pairs


def expand_pairs(pairs: list[dict], augment: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    templates = PROMPT_TEMPLATES[: max(1, min(augment, len(PROMPT_TEMPLATES)))]
    out = []
    for item in pairs:
        # 打乱模板顺序，增加多样性
        ts = templates[:]
        rng.shuffle(ts)
        for tmpl in ts:
            prompt = tmpl.format(q=item["question"])
            # 保证答案以句末结束，便于 EOS
            answer = item["answer"]
            out.append({"prompt": prompt, "answer": answer, "text": prompt + answer})
    rng.shuffle(out)
    return out


def export_replay_text(pairs: list[dict], path: str):
    """导出供继续预训练回放的纯文本（强化事实句式）。"""
    lines = []
    for item in pairs:
        q, a = item["question"], item["answer"]
        lines.append(f"问题：{q}\n答案：{a}\n")
        # 额外陈述句，利于「中国的首都是」类补全
        if "首都" in q and "北京" in a:
            lines.append("中国的首都是北京。\n")
            lines.append("中华人民共和国的首都是北京。\n")
            lines.append("北京是中国的首都。\n")
        lines.append(f"{q}{a}\n" if not q.endswith(("？", "?", "。", "：", ":")) else f"{q}\n{a}\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if not lines[-1].endswith("\n"):
            f.write("\n")
    print(f"✓ QA 回放文本已导出: {path} ({len(lines)} 段)")


def build_tokenizer(model_dir: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<pad>"
    tok.padding_side = "right"
    return tok


def tokenize_sft(examples, tokenizer, max_length: int, mask_prompt: bool):
    input_ids_list = []
    labels_list = []
    attn_list = []

    for prompt, answer in zip(examples["prompt"], examples["answer"]):
        # 答案后加 eos，便于停止
        full = prompt + answer + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_enc = tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        ids = full_enc["input_ids"]
        attn = full_enc["attention_mask"]

        if mask_prompt:
            labels = ids.copy()
            cut = min(len(prompt_ids), len(ids))
            # 至少保留最后 1 个 token 可学，避免全 mask
            cut = min(cut, max(0, len(ids) - 1))
            for i in range(cut):
                labels[i] = -100
        else:
            labels = ids.copy()

        input_ids_list.append(ids)
        labels_list.append(labels)
        attn_list.append(attn)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_list,
    }


def pick_precision() -> tuple[bool, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True
    if torch.cuda.is_available():
        return True, False
    return False, False


def generate_samples(model, tokenizer, device, prompts, max_new_tokens=40, temperature=0.3):
    model.eval()
    print("\n" + "=" * 60)
    print("问答生成测试")
    print("=" * 60)
    for prompt in prompts:
        # 统一成训练时见过的格式
        if not prompt.startswith("问题") and "###" not in prompt:
            text = f"问题：{prompt}\n答案："
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n[提示] {text!r}")
        print(f"[输出] {decoded}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mask_prompt = args.mask_prompt and not args.no_mask_prompt

    print("=" * 60)
    print("中文 GPT-2 问答 SFT")
    print("=" * 60)

    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"找不到模型目录: {args.model_dir}")

    tokenizer = build_tokenizer(args.model_dir)
    pairs = load_qa_pairs(args.qa_file)
    print(f"✓ 加载问答 {len(pairs)} 条 from {args.qa_file}")
    export_replay_text(pairs, args.replay_file)

    expanded = expand_pairs(pairs, args.augment, args.seed)
    print(f"✓ 模板扩写后样本数: {len(expanded)} (augment={args.augment})")

    if args.skip_train:
        model = GPT2LMHeadModel.from_pretrained(args.output_dir if Path(args.output_dir).exists() else args.model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        prompts = [args.prompt] if args.prompt else [
            "中国的首都是",
            "中国的首都是哪里？",
            "日本的首都是哪里？",
            "什么是深度学习？",
        ]
        generate_samples(model, tokenizer, device, prompts, args.max_new_tokens, args.temperature)
        return

    raw = Dataset.from_list(expanded)
    if args.eval_ratio > 0 and len(raw) >= 20:
        split = raw.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw, eval_raw = raw, None

    def _tok(batch):
        return tokenize_sft(batch, tokenizer, args.max_length, mask_prompt)

    train_ds = train_raw.map(_tok, batched=True, remove_columns=train_raw.column_names, desc="Tokenize train")
    eval_ds = None
    if eval_raw is not None:
        eval_ds = eval_raw.map(_tok, batched=True, remove_columns=eval_raw.column_names, desc="Tokenize eval")

    print(f"✓ 训练样本: {len(train_ds)}" + (f", 验证: {len(eval_ds)}" if eval_ds else ""))

    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    # 确保 pad 与 config 一致
    model.config.pad_token_id = tokenizer.pad_token_id

    fp16, bf16 = pick_precision()
    use_eval = eval_ds is not None and len(eval_ds) > 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.save_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        logging_first_step=True,
        seed=args.seed,
        remove_unused_columns=False,
    )

    # 动态 padding；labels 中 -100 需保留
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if use_eval else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\n开始 SFT ...")
    print(f"  model: {args.model_dir}")
    print(f"  output: {args.output_dir}")
    print(f"  lr={args.learning_rate}, epochs={args.epochs}, mask_prompt={mask_prompt}")
    print(f"  precision: fp16={fp16}, bf16={bf16}")

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if use_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"eval_loss={metrics.get('eval_loss')}")

    print(f"\n✓ SFT 模型已保存: {args.output_dir}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompts = [args.prompt] if args.prompt else [
        "中国的首都是",
        "中国的首都是哪里？",
        "上海是中国的首都吗？",
        "日本的首都是哪里？",
        "什么是深度学习？",
        "中国的国庆日是哪一天？",
    ]
    generate_samples(model, tokenizer, device, prompts, args.max_new_tokens, args.temperature)

    print("\n" + "=" * 60)
    print("SFT 完成。下一步可热启动续训：")
    print(
        "  python 04.train_gpt2.py --init-from gpt2-zh-sft-qa "
        "--output-dir gpt2-zh-sft-cpt --max-steps 30000 ..."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
