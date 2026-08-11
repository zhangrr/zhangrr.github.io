import re
import argparse
from pathlib import Path


DEFAULT_INPUT_DIR = "train_data_chunks"
DEFAULT_INPUT_FILE = "train_data.txt"
DEFAULT_OUTPUT_DIR = "train_data_cleaned_chunks"

def clean_text(text):
    """清洗文本数据"""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)

    # 移除 URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 移除电子邮件
    text = re.sub(r'\S+@\S+', '', text)

    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：、""''（）【】《》…—·]', '', text)

    # 去除首尾空格
    text = text.strip()

    return text

def parse_args():
    parser = argparse.ArgumentParser(description="Clean text data into chunked output files.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=50)
    return parser.parse_args()


def iter_input_files(input_path):
    path = Path(input_path)
    if path.is_dir():
        yield from sorted(path.glob("*.txt"))
    elif path.exists():
        yield path
    elif input_path == DEFAULT_INPUT_DIR and Path(DEFAULT_INPUT_FILE).exists():
        yield Path(DEFAULT_INPUT_FILE)
    else:
        raise FileNotFoundError(f"找不到输入路径: {input_path}")


def open_output_chunk(output_dir, chunk_index):
    path = output_dir / f"train_data_cleaned_{chunk_index:06d}.txt"
    return path, path.open("w", encoding="utf-8")


def process_files(input_files, output_dir, chunk_size, min_length=50):
    """分批清洗文件，避免生成过大的单个输出文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"开始处理，输出目录: {output_dir}")

    chunk_index = 0
    chunk_bytes = 0
    total_lines = 0
    valid_lines = 0
    chunk_path, chunk_file = open_output_chunk(output_dir, chunk_index)

    try:
        for input_file in input_files:
            print(f"处理 {input_file}...")
            with input_file.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    total_lines += 1
                    cleaned = clean_text(line)

                    if len(cleaned) < min_length:
                        continue

                    record = cleaned + "\n\n"
                    record_bytes = len(record.encode("utf-8"))
                    if chunk_bytes and chunk_bytes + record_bytes > chunk_size:
                        chunk_file.close()
                        print(f"写入 {chunk_path} ({chunk_bytes / 1024 / 1024:.2f} MB)")
                        chunk_index += 1
                        chunk_bytes = 0
                        chunk_path, chunk_file = open_output_chunk(output_dir, chunk_index)

                    chunk_file.write(record)
                    chunk_bytes += record_bytes
                    valid_lines += 1

                    if total_lines % 10000 == 0:
                        print(f"已处理 {total_lines} 行，保留 {valid_lines} 行...")
    finally:
        chunk_file.close()

    if chunk_bytes == 0 and chunk_path.exists():
        chunk_path.unlink()
    else:
        print(f"写入 {chunk_path} ({chunk_bytes / 1024 / 1024:.2f} MB)")

    print("\n处理完成！")
    print(f"总行数: {total_lines}")
    if total_lines:
        print(f"保留行数: {valid_lines} ({valid_lines / total_lines * 100:.1f}%)")
    else:
        print("保留行数: 0")

    total_size = sum(path.stat().st_size for path in output_dir.glob("*.txt"))
    print(f"输出总大小: {total_size / (1024*1024):.2f} MB")

# 使用示例
if __name__ == "__main__":
    args = parse_args()
    try:
        files = list(iter_input_files(args.input))
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        print("请先运行 01_load_dataset.py 生成分片数据")
    else:
        process_files(files, Path(args.output_dir), args.chunk_mb * 1024 * 1024, args.min_length)
