"""
Module: nodes/test_node.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Internal test node implementation.

Purpose:
    Provides a lightweight node used to verify extension loading and execution paths.
"""

class ALEXZTestNode:
    """ComfyUI test node used to verify extension loading and execution."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "text": ("STRING", {"default": "hello", "tooltip": "Тестовый текст."}),
                "value": ("INT", {"default": 1, "min": -999999, "max": 999999, "tooltip": "Тестовое число."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text_out", "value_out")
    FUNCTION = "run"
    CATEGORY = "utils/debug"

    def run(self, text, value):
        """Execute this helper/test function and return its outputs."""
        return (f"ALEXZ_TEST: {text}", int(value))
