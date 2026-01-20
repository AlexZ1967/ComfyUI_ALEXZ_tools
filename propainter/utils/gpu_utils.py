"""GPU memory management utilities."""

import logging
import torch
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def gpu_memory_manager(empty_cache: bool = True) -> Generator[None, None, None]:
    """
    Context manager for automatic GPU memory cleanup.
    
    Args:
        empty_cache: If True (default), call torch.cuda.empty_cache() on exit.
                    Set to False to improve throughput if OOM is not a concern.
    
    Usage:
        with gpu_memory_manager():
            # GPU operations (with cache cleanup)
            result = model(input)
        
        # For throughput-optimized scenario:
        with gpu_memory_manager(empty_cache=False):
            result = model(input)
    """
    try:
        yield
    finally:
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleaned up")


def cleanup_gpu(force: bool = True) -> None:
    """
    Explicitly cleanup GPU memory if available.
    
    Args:
        force: If True, always call empty_cache(). If False, may skip in throughput mode.
    """
    if force and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cleaned up")
