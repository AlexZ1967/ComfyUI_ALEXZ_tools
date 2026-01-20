from dataclasses import dataclass
import logging
import os
from pathlib import Path

import torch

from ..model.modules.flow_comp_raft import RAFT_bi
from ..model.propainter import InpaintGenerator
from ..model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model_cache import ModelCache
from .download_utils import download_model


logger = logging.getLogger(__name__)


@dataclass
class Models:
    raft_model: RAFT_bi
    flow_model: RecurrentFlowCompleteNet
    inpaint_model: InpaintGenerator


_WEIGHT_FILES = {
    "raft": "raft-things.pth",
    "flow": "recurrent_flow_completion.pth",
    "inpaint": "ProPainter.pth",
}
_PRETRAIN_MODEL_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def _get_weights_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(repo_root, "propainter", "weights")


def _resolve_weight_path(key: str) -> str:
    weights_dir = Path(_get_weights_dir())
    filename = _WEIGHT_FILES[key]
    path = weights_dir / filename
    if path.exists():
        return str(path)
    try:
        return download_model(_PRETRAIN_MODEL_URL, filename, weights_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Missing weight file: {path}. Place {filename} in {weights_dir} "
            "or allow auto-download."
        ) from exc


def load_raft_model(device: torch.device, model_path: str | None = None) -> RAFT_bi:
    """Loads the RAFT bi-directional model.
    
    Args:
        device: Target device (cpu/cuda)
        model_path: Path to model weights, auto-downloads if None
        
    Returns:
        Loaded RAFT model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info("Loading RAFT model...")
        if model_path is None:
            model_path = _resolve_weight_path("raft")
        raft_model = RAFT_bi(model_path, device)
        logger.info("RAFT model loaded successfully")
        return raft_model
    except Exception as e:
        logger.error(f"Failed to load RAFT model: {e}")
        raise RuntimeError(f"RAFT model loading failed: {e}") from e


def load_recurrent_flow_model(
    device: torch.device, model_path: str | None = None
) -> RecurrentFlowCompleteNet:
    """Loads the Recurrent Flow Completion Network model.
    
    Args:
        device: Target device (cpu/cuda)
        model_path: Path to model weights, auto-downloads if None
        
    Returns:
        Loaded flow completion model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info("Loading Recurrent Flow Completion model...")
        if model_path is None:
            model_path = _resolve_weight_path("flow")
        flow_model = RecurrentFlowCompleteNet(model_path)
        flow_model.to(device)
        flow_model.eval()
        logger.info("Flow completion model loaded successfully")
        return flow_model
    except Exception as e:
        logger.error(f"Failed to load flow completion model: {e}")
        raise RuntimeError(f"Flow completion model loading failed: {e}") from e


def load_inpaint_model(
    device: torch.device, model_path: str | None = None
) -> InpaintGenerator:
    """Loads the Inpaint Generator model.
    
    Args:
        device: Target device (cpu/cuda)
        model_path: Path to model weights, auto-downloads if None
        
    Returns:
        Loaded inpainting model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info("Loading ProPainter Inpainting model...")
        if model_path is None:
            model_path = _resolve_weight_path("inpaint")
        inpaint_model = InpaintGenerator(model_path=model_path).to(device)
        inpaint_model.eval()
        logger.info("Inpainting model loaded successfully")
        return inpaint_model
    except Exception as e:
        logger.error(f"Failed to load inpainting model: {e}")
        raise RuntimeError(f"Inpainting model loading failed: {e}") from e


def initialize_models(device: torch.device, use_half: str) -> Models:
    """
    Return initialized inference models with caching support.
    
    Models are cached per device and FP16 setting to avoid redundant loading.
    
    Args:
        device: Target device (cpu/cuda)
        use_half: "enable" or "disable" FP16 precision
        
    Returns:
        Models object with loaded and configured models
    """
    use_fp16 = use_half == "enable"
    
    weight_paths = {key: _resolve_weight_path(key) for key in _WEIGHT_FILES}

    # Try to get cached models
    cache = ModelCache()
    cached_models = cache.get_cached_models(device, use_fp16, weight_paths)
    
    if cached_models is not None:
        logger.info(f"Using cached models (device: {device}, fp16: {use_fp16})")
        return Models(
            cached_models.raft_model,
            cached_models.flow_model,
            cached_models.inpaint_model
        )
    
    logger.info(f"Loading models (device: {device}, fp16: {use_fp16})")
    
    # Load models
    raft_model = load_raft_model(device, model_path=weight_paths["raft"])
    flow_model = load_recurrent_flow_model(device, model_path=weight_paths["flow"])
    inpaint_model = load_inpaint_model(device, model_path=weight_paths["inpaint"])

    # Apply FP16 if requested (but not for RAFT on CPU)
    if use_fp16 and device != torch.device("cpu"):
        flow_model = flow_model.half()
        inpaint_model = inpaint_model.half()
        # RAFT is kept in FP32 due to numerical stability
        logger.info("Applied FP16 to flow and inpaint models")
    
    # Cache the models
    cache.set_cached_models(
        raft_model,
        flow_model,
        inpaint_model,
        device,
        use_fp16,
        weight_paths=weight_paths,
    )
    
    return Models(raft_model, flow_model, inpaint_model)
