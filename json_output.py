import json
import logging
import os


_LOGGER = logging.getLogger("JsonDisplayAndSave")


def _parse_json_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def _normalize_json_input(json_text):
    if isinstance(json_text, (list, tuple)):
        return [_parse_json_value(item) for item in json_text]
    return _parse_json_value(json_text)


class JsonDisplayAndSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "tooltip": "JSON строка для отображения и сохранения."}),
                "output_path": ("STRING", {"default": "output/transform.json", "tooltip": "Путь для сохранения JSON файла."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_pretty",)
    FUNCTION = "display"
    CATEGORY = "utils/json"
    OUTPUT_NODE = True

    def display(self, json_text, output_path):
        data = _normalize_json_input(json_text)
        pretty = json.dumps(data, ensure_ascii=True, indent=2)

        if not output_path or not str(output_path).strip():
            raise ValueError("output_path is required.")

        output_path = os.path.expanduser(str(output_path))
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(pretty)

        _LOGGER.info("Saved JSON to %s", output_path)

        return {"ui": {"text": [pretty]}, "result": (pretty,)}


_LOGGER.warning(
    "Loaded JsonDisplayAndSave. NODE_CLASS_MAPPINGS=%s",
    ["JsonDisplayAndSave"],
)
