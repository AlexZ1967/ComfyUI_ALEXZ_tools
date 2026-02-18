"""
Module: utils/module_browser/module/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Module metadata, identity, and information helpers.
"""

from .module_identity import (
    build_custom_module_aliases,
    canonical_custom_module_name,
    discover_custom_modules,
    normalize_module_token,
)
from .module_info import (
    cached_module_flags,
    resolve_module_info_uncached,
)
from .module_info_text import (
    module_local_readme_summary,
    sanitize_module_description,
)
from .module_update_state_ops import (
    comfyui_needs_update_now,
    count_custom_modules_need_update,
    count_custom_modules_unknown_update,
    module_needs_update_now,
)
from .node_classification_ops import (
    classify_by_relative_module,
    classify_by_source_path,
    fallback_annotation,
    module_root,
)
from .node_snapshot_ops import (
    build_node_snapshots,
    file_digest,
    node_source_file,
    relative_to_custom_roots,
)

__all__ = [
    "discover_custom_modules",
    "normalize_module_token",
    "build_custom_module_aliases",
    "canonical_custom_module_name",
    "cached_module_flags",
    "resolve_module_info_uncached",
    "module_local_readme_summary",
    "sanitize_module_description",
    "module_needs_update_now",
    "count_custom_modules_need_update",
    "count_custom_modules_unknown_update",
    "comfyui_needs_update_now",
    "classify_by_source_path",
    "classify_by_relative_module",
    "fallback_annotation",
    "module_root",
    "node_source_file",
    "relative_to_custom_roots",
    "file_digest",
    "build_node_snapshots",
]
