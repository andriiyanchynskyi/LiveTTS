import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List

from ..repositories.voice_repo import list_voices, get_voice_by_name, aggregate_languages, VoiceModel as RepoVoiceModel
from ..utils import json_load, json_save, load_voice_config


class VoiceListResponse(BaseModel):
    voices: List[RepoVoiceModel]
    total_count: int


def create_router(voices_dir, custom_ref_path, custom_state_path, zero_shot_base, supported_languages):
    router = APIRouter()
    logger = logging.getLogger(__name__)

    @router.get("/voices", response_model=VoiceListResponse)
    async def get_voices():
        voices = list_voices(voices_dir, custom_ref_path, custom_state_path, zero_shot_base, supported_languages)
        logger.info("[voices] listed voices: count=%d", len(voices))
        return VoiceListResponse(voices=voices, total_count=len(voices))

    @router.post("/voices/refresh")
    async def refresh_voices():
        voices = list_voices(voices_dir, custom_ref_path, custom_state_path, zero_shot_base, supported_languages)
        logger.info("[voices] refresh requested: found=%d", len(voices))
        return {"message": "Voice list refreshed", "voices_found": len(voices), "voices": [v.name for v in voices]}

    @router.get("/languages")
    async def get_languages():
        languages_list = aggregate_languages(voices_dir, supported_languages)
        logger.info("[voices] languages aggregated: count=%d", len(languages_list))
        return {"languages": languages_list, "total_count": len(languages_list)}

    # -------- Zero-shot: upload & clear --------
    @router.post("/custom-voice/upload")
    async def upload_custom_voice(file: UploadFile = File(...)):
        if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
            raise HTTPException(status_code=400, detail="Only WAV files are accepted")
        data = await file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20MB)")

        try:
            with open(custom_ref_path, "wb") as f:
                f.write(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

        meta = {"name": "Custom", "language": "en"}
        try:
            json_save(custom_state_path, meta)
        except Exception:
            pass
        logger.info("[voices] custom voice uploaded: bytes=%d name=%s", len(data), meta["name"])
        return {"success": True, "name": meta["name"]}

    @router.post("/custom-voice/clear")
    async def custom_voice_clear():
        try:
            if custom_ref_path.exists():
                custom_ref_path.unlink()
            if custom_state_path.exists():
                custom_state_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to clear zero-shot voice: {e}")
        logger.info("[voices] custom voice cleared")
        return {"success": True, "message": "Zero-shot voice cleared"}

    @router.get("/custom-voice/status")
    async def custom_voice_status():
        result = {
            "active": custom_ref_path.exists(),
            "name": (json_load(custom_state_path).get("name") if custom_state_path.exists() else None),
            "path": str(custom_ref_path) if custom_ref_path.exists() else None,
        }
        logger.info("[voices] custom voice status: active=%s name=%s", result["active"], result["name"])
        return result

    @router.get("/voices/{voice_name}", response_model=RepoVoiceModel)
    async def get_voice(voice_name: str):
        voices = list_voices(voices_dir, custom_ref_path, custom_state_path, zero_shot_base, supported_languages)
        voice = get_voice_by_name(voices, voice_name)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
        logger.info("[voices] fetched voice: name=%s is_zero_shot=%s", voice.name, getattr(voice, 'is_zero_shot', False))
        return voice

    @router.get("/voices/{voice_name}/languages")
    async def get_voice_languages(voice_name: str):
        voices = list_voices(voices_dir, custom_ref_path, custom_state_path, zero_shot_base, supported_languages)
        voice = get_voice_by_name(voices, voice_name)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
        if voice.is_zero_shot:
            result = {
                "voice_name": voice_name,
                "supported_languages": supported_languages,
                "fixed_language": None,
                "voice_language": voice.language,
                "can_change_language": True
            }
            logger.info("[voices] languages for zero-shot voice=%s: %d", voice_name, len(result["supported_languages"]))
            return result
        try:
            cfg = load_voice_config(voice.config_path)
            result = {
                "voice_name": voice_name,
                "supported_languages": cfg.get("languages", supported_languages),
                "fixed_language": cfg.get("language"),
                "voice_language": voice.language,
                "can_change_language": cfg.get("language") is None
            }
            logger.info("[voices] languages for voice=%s: %d (fixed=%s)", voice_name, len(result["supported_languages"]), result["fixed_language"])
            return result
        except Exception:
            result = {
                "voice_name": voice_name,
                "supported_languages": supported_languages,
                "fixed_language": None,
                "voice_language": voice.language,
                "can_change_language": True
            }
            logger.warning("[voices] fallback languages for voice=%s: %d", voice_name, len(result["supported_languages"]))
            return result

    return router


