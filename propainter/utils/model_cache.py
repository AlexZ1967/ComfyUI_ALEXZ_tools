"""Model caching system to avoid reloading models on every node execution."""

import torch
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional
import os

logger = logging.getLogger(__name__)


@dataclass
class CachedModels:
    """Container for cached models."""
    raft_model: Optional[torch.nn.Module] = None
    flow_model: Optional[torch.nn.Module] = None
    inpaint_model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    fp16_enabled: bool = False
    # Track weights hash to detect model changes
    weights_hash: Optional[str] = None


class ModelCache:
    """Singleton cache for loaded models to avoid redundant loading."""
    
    _instance: Optional['ModelCache'] = None
    _cache: CachedModels = CachedModels()
    
    def __new__(cls) -> 'ModelCache':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def _compute_weights_hash(weight_paths: dict[str, str]) -> str:
        """Compute hash of weight file paths and modification times.
        
        Args:
            weight_paths: Dict mapping model name to weight file path
            
        Returns:
            Hash string for cache validation
        """
        hash_input = []
        for name, path in weight_paths.items():
            if os.path.exists(path):
                # Include path and mtime to detect file changes
                mtime = os.path.getmtime(path)
                hash_input.append(f"{name}:{path}:{mtime}")
            else:
                hash_input.append(f"{name}:{path}:missing")
        
        hash_str = "|".join(hash_input)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def get_cached_models(
        self, 
        device: torch.device, 
        use_fp16: bool,
        weight_paths: Optional[dict[str, str]] = None
    ) -> Optional[CachedModels]:
        """
        Get cached models if they exist and match requirements.
        
        Args:
            device: Target device (cpu/cuda)
            use_fp16: Whether FP16 precision is required
            weight_paths: Dict mapping model name to weight file path for validation
            
        Returns:
            Cached models if available and matching, None otherwise
        """
        # Check if cache is valid
        if (self._cache.raft_model is None or 
            self._cache.flow_model is None or 
            self._cache.inpaint_model is None):
            return None
        
        # Validate weights hash if provided (detects model file changes)
        if weight_paths is not None:
            current_hash = self._compute_weights_hash(weight_paths)
            if self._cache.weights_hash != current_hash:
                logger.debug("Weight files changed, invalidating cache")
                self.clear()
                return None
        
        # Check if device matches
        if self._cache.device != device:
            return None
        
        # Check if FP16 setting matches
        if self._cache.fp16_enabled != use_fp16:
            return None
        
        return self._cache
    
    def set_cached_models(
        self, 
        raft_model: torch.nn.Module,
        flow_model: torch.nn.Module,
        inpaint_model: torch.nn.Module,
        device: torch.device,
        use_fp16: bool,
        weight_paths: Optional[dict[str, str]] = None
    ) -> None:
        """
        Store models in cache.
        
        Args:
            raft_model: RAFT optical flow model
            flow_model: Flow completion model
            inpaint_model: ProPainter inpainting model
            device: Target device
            use_fp16: Whether FP16 is enabled
            weight_paths: Dict mapping model name to weight file path (for cache validation)
        """
        weights_hash = None
        if weight_paths is not None:
            weights_hash = self._compute_weights_hash(weight_paths)
        
        self._cache = CachedModels(
            raft_model=raft_model,
            flow_model=flow_model,
            inpaint_model=inpaint_model,
            device=device,
            fp16_enabled=use_fp16,
            weights_hash=weights_hash
        )
    
    def clear(self) -> None:
        """Clear all cached models and free memory."""
        if self._cache.raft_model is not None:
            self._cache.raft_model.cpu()
            del self._cache.raft_model
        
        if self._cache.flow_model is not None:
            self._cache.flow_model.cpu()
            del self._cache.flow_model
        
        if self._cache.inpaint_model is not None:
            self._cache.inpaint_model.cpu()
            del self._cache.inpaint_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._cache = CachedModels()
        logger.debug("Model cache cleared")
    
    def is_cached(self) -> bool:
        """Check if any models are cached."""
        return self._cache.raft_model is not None
