import json
import logging
import os


_LOGGER = logging.getLogger("JsonDisplayAndSave")


class AnyType(str):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


any_type = AnyType("*")


def _format_json_text(value):
    text = "{}"
    if value is None:
        return text
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=True, indent=2)
        except Exception:
            return json.dumps({"raw": value}, ensure_ascii=True, indent=2)
    if isinstance(value, list):
        parsed_items = [_parse_json_string(item) for item in value]
        try:
            return json.dumps(parsed_items, ensure_ascii=True, indent=2)
        except Exception:
            return json.dumps({"raw": value}, ensure_ascii=True, indent=2)
    parsed = _parse_json_string(value)
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=True, indent=2)
    return str(parsed)


def _parse_json_string(value):
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except Exception:
        return value
    if isinstance(parsed, str):
        inner = parsed.strip()
        if (inner.startswith("{") and inner.endswith("}")) or (
            inner.startswith("[") and inner.endswith("]")
        ):
            try:
                return json.loads(inner)
            except Exception:
                return parsed
    return parsed


def _unwrap_singleton(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


class JsonDisplayAndSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": (any_type, {"tooltip": "JSON строка или объект для отображения и сохранения."}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "tooltip": "Путь для сохранения JSON файла (необязательно)."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_pretty",)
    FUNCTION = "display"
    CATEGORY = "utils/json"
    OUTPUT_NODE = True

    def display(self, json_text, output_path=None):
        text = _format_json_text(_unwrap_singleton(json_text))

        output_value = _unwrap_singleton(output_path)
        if output_value and str(output_value).strip():
            output_path = os.path.expanduser(str(output_value))
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, "transform.json")
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(text)

            _LOGGER.info("Saved JSON to %s", output_path)

        return {"ui": {"text": [text]}, "result": (text,)}


_LOGGER.warning(
    "Loaded JsonDisplayAndSave. NODE_CLASS_MAPPINGS=%s",
    ["JsonDisplayAndSave"],
)
