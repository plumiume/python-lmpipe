# pyright: reportUnusedImport=false

import textwrap

try:
    import torch
except ImportError:
    raise ImportError(textwrap.dedent("""
        MMPose plugin requires PyTorch to be installed.
        Since GPU versions vary by environment (CUDA, ROCm, etc.),
        please install the appropriate PyTorch version manually:

        For CUDA:
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        For ROCm (AMD GPU):
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

        For CPU only:
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

        For more installation options, visit: https://pytorch.org/get-started/locally/
        """).strip())

