"""CUDA/cuDNN optimization utilities for performance tuning."""

import torch
import logging

logger = logging.getLogger(__name__)


def configure_cudnn(benchmark: bool = True, allow_tf32: bool = True) -> None:
    """
    Configure cuDNN settings for performance.
    
    Args:
        benchmark: If True, enable cuDNN.benchmark for faster convolutions
                  (may produce non-deterministic results for variable input sizes)
        allow_tf32: If True, allow TF32 tensor operations (faster but lower precision)
    
    Note:
        These settings trade determinism/stability for speed.
        Use benchmark=False for reproducible results with fixed input sizes.
        Use allow_tf32=False for strict FP32 precision.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cuDNN settings skipped")
        return
    
    torch.backends.cudnn.benchmark = benchmark
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    
    mode = "optimized" if benchmark else "deterministic"
    precision = "TF32" if allow_tf32 else "FP32"
    logger.debug(f"cuDNN configured: benchmark={mode}, precision={precision}")


def reset_cudnn_defaults() -> None:
    """Reset cuDNN settings to PyTorch defaults."""
    if not torch.cuda.is_available():
        return
    
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    logger.debug("cuDNN reset to defaults")
