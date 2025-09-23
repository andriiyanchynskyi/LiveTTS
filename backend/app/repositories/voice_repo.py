from pathlib import Path
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, validator

from ..utils.core import load_voice_config, json_load


class VoiceModel(BaseModel):
    name: str
    language: str
    voice_path: str
    config_path: str
    vocab_path: str
    m_path: str
    available: bool = True
    m_supported_languages: List[str] = []
    fixed_language: Optional[str] = None
    is_zero_shot: bool = False
    base_model: Optional[str] = None

    @validator('voice_path', 'config_path', 'vocab_path', 'm_path')
    def _path_must_exist(cls, v: str):
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


def voices_from_fs(voices_dir: Path) -> List[VoiceModel]:
    voices: List[VoiceModel] = []
    if not voices_dir.exists():
        return voices
    for lang_dir in voices_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        for voice_dir in lang_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            cfg = voice_dir / "config.json"
            mdl = voice_dir / "model.pth"
            voc = voice_dir / "vocab.json"
            ref = voice_dir / "reference.wav"
            if not (cfg.exists() and mdl.exists() and voc.exists() and ref.exists()):
                continue
            try:
                voice_cfg = load_voice_config(str(cfg))
                m_supported = voice_cfg.get("languages", [])
                fixed_lang = voice_cfg.get("language")
            except Exception:
                m_supported = []
                fixed_lang = None
            voices.append(
                VoiceModel(
                    name=voice_dir.name,
                    language=lang_dir.name,
                    voice_path=str(ref),
                    config_path=str(cfg),
                    vocab_path=str(voc),
                    m_path=str(mdl),
                    available=True,
                    m_supported_languages=m_supported,
                    fixed_language=fixed_lang,
                    is_zero_shot=False,
                )
            )
    return voices


def zero_shot_voice_entry(
    custom_ref_path: Path,
    custom_state_path: Path,
    zero_shot_base: Dict[str, str],
    supported_languages: List[str]
) -> Optional[VoiceModel]:
    if not custom_ref_path.exists():
        return None
    meta = {}
    if custom_state_path.exists():
        try:
            meta = json_load(custom_state_path)
        except Exception:
            meta = {}
    base_cfg = zero_shot_base["config_path"]
    base_m = zero_shot_base["m_path"]
    base_v = zero_shot_base["vocab_path"]
    return VoiceModel(
        name=meta.get("name", "Custom"),
        language=meta.get("language", "en"),
        voice_path=str(custom_ref_path),
        config_path=base_cfg,
        vocab_path=base_v,
        m_path=base_m,
        available=True,
        m_supported_languages=supported_languages,
        fixed_language=None,
        is_zero_shot=True,
        base_model=zero_shot_base.get("name")
    )


def list_voices(
    voices_dir: Path,
    custom_ref_path: Path,
    custom_state_path: Path,
    zero_shot_base: Dict[str, str],
    supported_languages: List[str]
) -> List[VoiceModel]:
    voices = voices_from_fs(voices_dir)
    zs = zero_shot_voice_entry(custom_ref_path, custom_state_path, zero_shot_base, supported_languages)
    if zs:
        voices = [zs] + voices
    return voices


def aggregate_languages(voices_dir: Path, fallback_languages: List[str]) -> List[str]:
    langs = set()
    for v in voices_from_fs(voices_dir):
        try:
            cfg = load_voice_config(v.config_path)
            langs.update([l.lower() for l in cfg.get("languages", fallback_languages)])
        except Exception:
            langs.update([l.lower() for l in fallback_languages])
    if not langs:
        langs.update([l.lower() for l in fallback_languages])
    return sorted(langs)


def get_voice_by_name(voices: List[VoiceModel], voice_name: str) -> Optional[VoiceModel]:
    return next((v for v in voices if v.name == voice_name), None)


