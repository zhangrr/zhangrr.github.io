import os
import argparse
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from pathlib import Path


DEFAULT_CHUNK_DIR = "train_data_cleaned_chunks"
DEFAULT_CLEANED_FILE = "train_data_cleaned.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from cleaned text chunks.")
    parser.add_argument("--input", default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--output-dir", default="tokenized_data")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--max-lines", type=int, default=5000000)
    parser.add_argument("--max-chars-per-line", type=int, default=800)
    parser.add_argument("--line-stride", type=int, default=5)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=100)
    return parser.parse_args()


def resolve_input_files(input_path):
    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if files:
            return files

    if path.is_file():
        return [path]

    if input_path == DEFAULT_CHUNK_DIR and Path(DEFAULT_CLEANED_FILE).is_file():
        return [Path(DEFAULT_CLEANED_FILE)]

    raise FileNotFoundError(f"找不到训练数据: {input_path}")


def iter_text_batches(paths, batch_size, max_lines, max_chars_per_line, line_stride):
    batch = []
    seen_lines = 0
    used_lines = 0
    for path in paths:
        print(f"读取训练数据: {path}")
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                seen_lines += 1
                if line_stride > 1 and seen_lines % line_stride != 0:
                    continue

                text = line.strip()
                if not text:
                    continue

                if max_chars_per_line > 0 and len(text) > max_chars_per_line:
                    text = text[:max_chars_per_line]

                batch.append(text)
                used_lines += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

                if max_lines > 0 and used_lines >= max_lines:
                    if batch:
                        yield batch
                    print(f"训练样本达到上限: {used_lines} 行，已扫描 {seen_lines} 行")
                    return

    if batch:
        yield batch
    print(f"训练样本数: {used_lines} 行，已扫描 {seen_lines} 行")


class BPE_token(object):

    def __init__(self):
        self.tokenizer = Tokenizer(BPE())

        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])

        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()


    def bpe_train(
        self,
        paths,
        batch_size=200,
        vocab_size=50000,
        max_lines=5000000,
        max_chars_per_line=800,
        line_stride=5,
        min_frequency=2,
        max_token_length=100,
    ):

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet(),
            max_token_length=max_token_length,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>"
            ]
        )

        self.tokenizer.train_from_iterator(
            iter_text_batches(
                paths,
                batch_size,
                max_lines,
                max_chars_per_line,
                line_stride,
            ),
            trainer=trainer,
        )


    def save_tokenizer(self, location):

        if not os.path.exists(location):
            os.makedirs(location)

        self.tokenizer.save(
            os.path.join(location, "tokenizer.json")
        )


if __name__ == "__main__":
    args = parse_args()
    paths = resolve_input_files(args.input)
    print(f"训练文件数: {len(paths)}")
    print(
        "训练限制: "
        f"max_lines={args.max_lines}, "
        f"max_chars_per_line={args.max_chars_per_line}, "
        f"line_stride={args.line_stride}, "
        f"min_frequency={args.min_frequency}, "
        f"max_token_length={args.max_token_length}"
    )

    tokenizer = BPE_token()
    tokenizer.bpe_train(
        paths,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        max_lines=args.max_lines,
        max_chars_per_line=args.max_chars_per_line,
        line_stride=args.line_stride,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
    )
    tokenizer.save_tokenizer(args.output_dir)
