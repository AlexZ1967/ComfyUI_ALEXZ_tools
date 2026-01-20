import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..model import e2fgvi as e2fgvi_model
from ..model import e2fgvi_hq as e2fgvi_hq_model
from .download_utils import ensure_file


logger = logging.getLogger(__name__)


_WEIGHTS = {
    "e2fgvi": ("E2FGVI-CVPR22.pth", "1tNJMTJ2gmWdIXJoHVi5-H504uImUiJW9"),
    "e2fgvi_hq": ("E2FGVI-HQ-CVPR22.pth", "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"),
}


def _get_weights_dir() -> Path:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return Path(repo_root) / "e2fgvi" / "weights"


def _load_weights(model_name: str) -> str:
    filename, file_id = _WEIGHTS[model_name]
    weights_dir = _get_weights_dir()
    path = weights_dir / filename
    return ensure_file(path, file_id=file_id)


@dataclass
class CachedModel:
    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    fp16: bool = False
    model_name: Optional[str] = None


_CACHE = CachedModel()


def load_model(model_name: str, device: torch.device, fp16: bool) -> torch.nn.Module:
    if (
        _CACHE.model is not None
        and _CACHE.device == device
        and _CACHE.fp16 == fp16
        and _CACHE.model_name == model_name
    ):
        return _CACHE.model

    if model_name == "e2fgvi_hq":
        model = e2fgvi_hq_model.InpaintGenerator()
    else:
        model = e2fgvi_model.InpaintGenerator()

    weights_path = _load_weights(model_name)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    if fp16 and device.type == "cuda":
        model = model.half()

    _CACHE.model = model
    _CACHE.device = device
    _CACHE.fp16 = fp16
    _CACHE.model_name = model_name
    logger.info("Loaded E2FGVI model %s from %s", model_name, weights_path)
    return model
