#!/usr/bin/env python3
"""Generate before/after examples for Color Match guide."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _load_color_match_module(repo_root: Path):
    pkg_name = "ComfyUI_ALEXZ_tools"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[pkg_name] = pkg
    if f"{pkg_name}.nodes" not in sys.modules:
        nodes_pkg = types.ModuleType(f"{pkg_name}.nodes")
        nodes_pkg.__path__ = [str(repo_root / "nodes")]
        sys.modules[f"{pkg_name}.nodes"] = nodes_pkg
    if f"{pkg_name}.utils" not in sys.modules:
        utils_pkg = types.ModuleType(f"{pkg_name}.utils")
        utils_pkg.__path__ = [str(repo_root / "utils")]
        sys.modules[f"{pkg_name}.utils"] = utils_pkg

    module_name = f"{pkg_name}.nodes.image_color_match"
    module_path = repo_root / "nodes" / "image_color_match.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _save_tensor_image(t: torch.Tensor, path: Path):
    arr = (torch.clamp(t, 0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    Image.fromarray(arr, mode="RGB").save(path)


def _load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return arr.unsqueeze(0)


def _make_synthetic_pair(height: int = 192, width: int = 320):
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=height),
        torch.linspace(0.0, 1.0, steps=width),
        indexing="ij",
    )

    base_r = 0.15 + 0.70 * xx
    base_g = 0.12 + 0.65 * yy
    base_b = 0.10 + 0.70 * (1.0 - xx * yy)
    image = torch.stack([base_r, base_g, base_b], dim=-1)

    ring = (((xx - 0.52) ** 2 + (yy - 0.47) ** 2) < 0.09).float()
    image[..., 0] = torch.clamp(image[..., 0] + ring * 0.20, 0.0, 1.0)
    image[..., 1] = torch.clamp(image[..., 1] - ring * 0.08, 0.0, 1.0)
    image[..., 2] = torch.clamp(image[..., 2] - ring * 0.15, 0.0, 1.0)

    bars = (((xx * 20.0).floor() % 2.0) == 0).float() * (((yy > 0.72) & (yy < 0.90)).float())
    image[..., 0] = torch.clamp(image[..., 0] * (1.0 - 0.12 * bars), 0.0, 1.0)
    image[..., 1] = torch.clamp(image[..., 1] * (1.0 + 0.10 * bars), 0.0, 1.0)

    ref = image.clone()
    ref[..., 0] = torch.clamp(ref[..., 0] * 0.76 + 0.16, 0.0, 1.0)
    ref[..., 1] = torch.clamp(ref[..., 1] * 0.97 + 0.02, 0.0, 1.0)
    ref[..., 2] = torch.clamp(ref[..., 2] * 1.16 + 0.01, 0.0, 1.0)

    vignette = torch.clamp(1.0 - 0.85 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2), 0.72, 1.05)
    ref = torch.clamp(ref * vignette.unsqueeze(-1), 0.0, 1.0)
    return image.unsqueeze(0), ref.unsqueeze(0)


def _run_case(node, out_dir: Path, image: torch.Tensor, reference: torch.Tensor, case_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_tensor_image(image[0], out_dir / "before_image.png")
    _save_tensor_image(reference[0], out_dir / "reference.png")

    presets = [
        "mean_std",
        "linear",
        "tone_curve",
        "adain",
        "optimal_transport",
        "lab_cdf",
        "oklab_cdf",
        "auto_optimal",
        "perceptual_vgg_fast",
    ]

    summary = {"case": case_name, "status": "ok", "presets": {}}
    for preset in presets:
        try:
            out, payload = node.match(
                reference,
                image,
                preset,
                compute_quality_metrics=False,
                quality_metrics_mode="off",
                strength=1.0,
            )
            out_path = out_dir / f"after_{preset}.png"
            _save_tensor_image(out[0, :, :, :3], out_path)
            summary["presets"][preset] = {
                "ok": True,
                "mode": json.loads(payload[0]).get("mode"),
                "image": str(out_path.name),
            }
        except Exception as exc:
            summary["presets"][preset] = {"ok": False, "error": str(exc)}

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated examples in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Color Match guide examples.")
    parser.add_argument("--case", default="case01", help="Case output directory name under color_match_examples/")
    parser.add_argument("--image", default="", help="Path to source image (image to be color-corrected).")
    parser.add_argument("--reference", default="", help="Path to reference image.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(__file__).resolve().parent / args.case

    mod = _load_color_match_module(repo_root)
    node = mod.ImageColorMatchToReference()

    if args.image and args.reference:
        image = _load_image_tensor(Path(args.image))
        reference = _load_image_tensor(Path(args.reference))
    else:
        image, reference = _make_synthetic_pair()

    _run_case(node, out_dir, image, reference, args.case)


if __name__ == "__main__":
    main()
