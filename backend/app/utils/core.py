import json
from pathlib import Path
from typing import Optional
from fastapi import HTTPException


def json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_voice_config(config_path: str) -> dict:
    try:
        return json_load(Path(config_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read voice config: {e}")


def validate_language(lang: Optional[str], voice_cfg: dict, fallback_languages: list) -> str:
    supported = [l.lower() for l in voice_cfg.get("languages", fallback_languages)]
    eff = (lang or "en").strip().lower()
    if eff not in supported:
        eff = supported[0]
    return eff


def get_inference_param(param_name: str, voice_cfg: dict, defaults: dict):
    value = voice_cfg.get(param_name)
    if value is None or value == '' or (isinstance(value, (int, float)) and value <= 0):
        return defaults[param_name]
    return value


