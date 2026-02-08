class ALEXZTestNode:
    @classmethod
    def INPUT_TYPES(cls):
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
        return (f"ALEXZ_TEST: {text}", int(value))
