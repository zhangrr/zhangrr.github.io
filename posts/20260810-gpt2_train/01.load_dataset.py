import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Load Wikipedia data into chunked text files.")
    parser.add_argument("--output-dir", default="train_data_chunks")
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--cache-dir", default="./hf_cache")
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def open_chunk(output_dir, chunk_index):
    path = output_dir / f"train_data_{chunk_index:06d}.txt"
    return path, path.open("w", encoding="utf-8")


def save_chunks(dataset, output_dir, chunk_size, min_length):
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_index = 0
    chunk_bytes = 0
    total_items = 0
    kept_items = 0
    chunk_path, chunk_file = open_chunk(output_dir, chunk_index)

    try:
        for item in dataset:
            total_items += 1
            text = (item.get("text") or "").strip()
            if len(text) < min_length:
                continue

            record = text + "\n\n"
            record_bytes = len(record.encode("utf-8"))
            if chunk_bytes and chunk_bytes + record_bytes > chunk_size:
                chunk_file.close()
                print(f"写入 {chunk_path} ({chunk_bytes / 1024 / 1024:.2f} MB)")
                chunk_index += 1
                chunk_bytes = 0
                chunk_path, chunk_file = open_chunk(output_dir, chunk_index)

            chunk_file.write(record)
            chunk_bytes += record_bytes
            kept_items += 1

            if total_items % 10000 == 0:
                print(f"已读取 {total_items} 条，保留 {kept_items} 条...")
    finally:
        chunk_file.close()

    if chunk_bytes == 0 and chunk_path.exists():
        chunk_path.unlink()
    else:
        print(f"写入 {chunk_path} ({chunk_bytes / 1024 / 1024:.2f} MB)")

    print("\n加载完成！")
    print(f"总样本数: {total_items}")
    print(f"保留样本数: {kept_items}")
    print(f"输出目录: {output_dir}")


def main():
    args = parse_args()
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    print(dataset)
    chunk_size = args.chunk_mb * 1024 * 1024
    save_chunks(dataset, Path(args.output_dir), chunk_size, args.min_length)


if __name__ == "__main__":
    main()
