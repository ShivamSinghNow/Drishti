from __future__ import annotations

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm is not available to PyTorch. Run this on the AMD MI300X ROCm host.")
        print(f"torch version: {torch.__version__}")
        print(f"ROCm version reported by torch: {getattr(torch.version, 'hip', None)}")
        return 1

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_gb = properties.total_memory / (1024**3)

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"GPU device name: {torch.cuda.get_device_name(device_index)}")
    print(f"Total GPU memory: {total_gb:.2f} GB")
    print(f"ROCm version: {getattr(torch.version, 'hip', None) or 'not reported by torch'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
