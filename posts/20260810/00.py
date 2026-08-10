import torch
import transformers
import datasets
import tokenizers

print("=" * 50)
print("环境检查")
print("=" * 50)

# PyTorch
print(f"✓ PyTorch 版本: {torch.__version__}")
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")
    print(f"✓ GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# HuggingFace
print(f"✓ Transformers 版本: {transformers.__version__}")
print(f"✓ Datasets 版本: {datasets.__version__}")
print(f"✓ Tokenizers 版本: {tokenizers.__version__}")

# 测试简单计算
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print(f"✓ GPU 计算测试通过")

print("=" * 50)
print("环境配置完成！可以开始训练。")
print("=" * 50)
