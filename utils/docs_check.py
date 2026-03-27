#!/usr/bin/env python3
"""
Module: utils/docs_check.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Documentation consistency checker.

Purpose:
    Validates README/guides against current node input/output definitions and reports mismatches.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

NODE_DOCS = [
    {
        "file": "nodes/image_prepare.py",
        "class": "ImagePrepareForQwenEditOutpaint",
        "guide": "guides/GUIDE_IMAGE_PREP.md",
        "readme_heading": "## Image Prepare for QwenEdit Outpaint",
    },
    {
        "file": "nodes/image_align.py",
        "class": "ImageAlignOverlayToBackground",
        "guide": "guides/GUIDE_ALIGN.md",
        "readme_heading": "## Align Overlay To Background",
    },
    {
        "file": "nodes/image_color_match.py",
        "class": "ImageColorMatchToReference",
        "guide": "guides/GUIDE_COLOR_MATCH_DETAILED.md",
        "readme_heading": "## Color Match To Reference",
    },
    {
        "file": "nodes/image_look_match.py",
        "class": "ImageLookMatchResolve",
        "guide": "guides/GUIDE_LOOK_MATCH.md",
        "readme_heading": "## Look Match Resolve",
    },
    {
        "file": "nodes/image_look_match.py",
        "class": "ImageLookMatchNukeBuild",
        "guide": "guides/GUIDE_LOOK_MATCH.md",
        "readme_heading": "## Look Match Nuke Build",
    },
    {
        "file": "nodes/image_look_match.py",
        "class": "ImageLookMatchNukeApply",
        "guide": "guides/GUIDE_LOOK_MATCH.md",
        "readme_heading": "## Look Match Nuke Apply",
    },
    {
        "file": "nodes/image_seam_match.py",
        "class": "ImageSeamMatchToReference",
        "guide": "guides/GUIDE_SEAM_MATCH.md",
        "readme_heading": "## Seam Match To Reference",
    },
    {
        "file": "nodes/video_frame_match.py",
        "class": "VideoFrameMatch",
        "guide": "guides/GUIDE_VIDEO_FRAME_MATCH.md",
        "readme_heading": "## Find Closest Video Frame",
    },
    {
        "file": "nodes/video_cut_match.py",
        "class": "VideoCutMatch",
        "guide": "guides/GUIDE_VIDEO_CUT_MATCH.md",
        "readme_heading": "## Match Video Cut Point",
    },
    {
        "file": "nodes/image_difference.py",
        "class": "ImageDifference",
        "guide": "guides/GUIDE_IMAGE_DIFFERENCE.md",
        "readme_heading": "## Image Difference",
    },
    {
        "file": "nodes/qr_code_generate.py",
        "class": "GenerateQRCode",
        "guide": "guides/GUIDE_QR_CODE.md",
        "readme_heading": "## Generate QR Code",
    },
    {
        "file": "nodes/image_download_dzi_tiles.py",
        "class": "ImageDownloadDZITiles",
        "guide": "guides/GUIDE_DZI_TILES_DOWNLOAD.md",
        "readme_heading": "## Download DZI Tiles Image",
    },
    {
        "file": "nodes/image_download_dzi_tiles.py",
        "class": "ImageDownloadDZITilesBatchSave",
        "guide": "guides/GUIDE_DZI_TILES_DOWNLOAD.md",
        "readme_heading": "## Download DZI Tiles Batch Save",
    },
    {
        "file": "nodes/image_download_iiif.py",
        "class": "ImageDownloadIIIFImage",
        "guide": "guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md",
        "readme_heading": "## Download IIIF Image",
    },
    {
        "file": "nodes/trove_search_ids.py",
        "class": "SearchTroveImageIDs",
        "guide": "guides/GUIDE_TROVE_SEARCH_IDS.md",
        "readme_heading": "## Search Trove Image IDs",
    },
    {
        "file": "nodes/image_scopes.py",
        "class": "ImageWaveformScope",
        "guide": "guides/GUIDE_IMAGE_WAVEFORM.md",
        "readme_heading": "## Image Waveform Scope",
    },
    {
        "file": "nodes/image_scopes.py",
        "class": "ImageHistogramScope",
        "guide": "guides/GUIDE_IMAGE_HISTOGRAM.md",
        "readme_heading": "## Image Histogram Scope",
    },
    {
        "file": "nodes/video_inpaint.py",
        "class": "VideoInpaintWatermark",
        "guide": "guides/GUIDE_VIDEO_INPAINT.md",
        "readme_heading": "## Remove Static Watermark from Video",
    },
    {
        "file": "nodes/json_output.py",
        "class": "JsonDisplayAndSave",
        "guide": "guides/GUIDE_JSON.md",
        "readme_heading": "## Show/Save JSON",
    },
]

TEMPLATE_SECTIONS = [
    "## Назначение",
    "## Когда использовать",
    "## Минимальный сценарий (3 шага)",
    "## Параметры",
    "## Decision helper",
    "## Интерпретация выходов",
    "## Типовые ошибки и решения",
    "## Производительность",
]

EXTRA_GUIDES = [
    "guides/GUIDE_COLOR_MATCH.md",
]


def _literal_str_tuple(node):
    """Normalize AST literal nodes into a tuple of strings."""
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for el in node.elts:
            if isinstance(el, ast.Constant) and isinstance(el.value, str):
                out.append(el.value)
        return out
    return []


def _find_class(tree, class_name):
    """Find a class node by name inside parsed Python AST."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _parse_return_names(class_node):
    """Extract RETURN_NAMES tuple values from a class AST definition."""
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "RETURN_NAMES":
                    return _literal_str_tuple(stmt.value)
    return []


def _extract_dict_keys(dict_node):
    """Extract top-level dictionary keys from source snippet."""
    if not isinstance(dict_node, ast.Dict):
        return []
    out = []
    for k in dict_node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            out.append(k.value)
    return out


def _parse_input_keys(class_node):
    """Parse INPUT_TYPES keys and optional parameter keys from class source."""
    required = []
    optional = []
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "INPUT_TYPES":
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    root = sub.value
                    for k, v in zip(root.keys, root.values):
                        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                            continue
                        if k.value == "required":
                            required = _extract_dict_keys(v)
                        elif k.value == "optional":
                            optional = _extract_dict_keys(v)
                    return required, optional
    return required, optional


def _read(path: Path):
    """Read text file content using UTF-8 encoding."""
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str):
    """Extract text block between section headings in markdown files."""
    idx = text.find(heading)
    if idx < 0:
        return ""
    m = re.search(r"\n## ", text[idx + len(heading):])
    if not m:
        return text[idx:]
    end = idx + len(heading) + m.start() + 1
    return text[idx:end]


def main():
    """Run docs consistency checks and print a pass/fail report."""
    issues = []
    readme_text = _read(README)

    for spec in NODE_DOCS:
        py_path = ROOT / spec["file"]
        guide_path = ROOT / spec["guide"]
        if not py_path.exists():
            issues.append(f"missing source file: {spec['file']}")
            continue
        if not guide_path.exists():
            issues.append(f"missing guide file: {spec['guide']}")
            continue

        tree = ast.parse(_read(py_path), filename=str(py_path))
        cls = _find_class(tree, spec["class"])
        if cls is None:
            issues.append(f"class not found: {spec['class']} in {spec['file']}")
            continue

        required, optional = _parse_input_keys(cls)
        outputs = _parse_return_names(cls)

        guide_text = _read(guide_path)

        for header in TEMPLATE_SECTIONS:
            if header not in guide_text:
                issues.append(f"{spec['guide']}: missing template section '{header}'")

        for key in required:
            if f"`{key}`" not in guide_text:
                issues.append(f"{spec['guide']}: missing required input `{key}`")
        for key in outputs:
            if f"`{key}`" not in guide_text:
                issues.append(f"{spec['guide']}: missing output `{key}`")

        readme_section = _section(readme_text, spec["readme_heading"])
        if not readme_section:
            issues.append(f"README missing section: {spec['readme_heading']}")
            continue

        if spec["guide"] not in readme_section:
            issues.append(f"README section '{spec['readme_heading']}' missing guide link {spec['guide']}")

        for key in outputs:
            if f"`{key}`" not in readme_section:
                issues.append(f"README section '{spec['readme_heading']}' missing output `{key}`")

    for guide_name in EXTRA_GUIDES:
        guide_path = ROOT / guide_name
        if not guide_path.exists():
            issues.append(f"missing extra guide file: {guide_name}")
            continue
        guide_text = _read(guide_path)
        for header in TEMPLATE_SECTIONS:
            if header not in guide_text:
                issues.append(f"{guide_name}: missing template section '{header}'")

    if issues:
        print("docs-check: FAILED")
        for item in issues:
            print(f" - {item}")
        return 1

    print("docs-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
