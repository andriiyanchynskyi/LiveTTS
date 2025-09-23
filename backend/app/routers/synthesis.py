import uuid
import os
import asyncio
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import soundfile as sf

from ..enhancements import apply_enhancements
from ..utils import (
    get_inference_param, 
    validate_language, 
    load_voice_config,
    should_enable_text_splitting, 
    validate_text_splitting_configuration,
 )


def create_router(
    list_voices_func,
    voices_dir: Path,
    output_dir: Path,
    custom_ref_path: Path,
    zero_shot_base: Dict[str, str],
    supported_languages: list,
    load_xtts_func,
    logger,
):
    router = APIRouter()

    INFERENCE_TIMEOUT_SEC = float(os.getenv("INFERENCE_TIMEOUT_SEC", "60"))

    INFERENCE_DEFAULTS = {
        "temperature": 0.75,
        "length_penalty": 1.0,
        "repetition_penalty": 5.0,
        "top_k": 50,
        "top_p": 0.85,
        "speed": 1.0,
    }

    def build_params(voice_cfg: dict, req_speed: float | None) -> dict:
        params = {k: get_inference_param(k, voice_cfg, INFERENCE_DEFAULTS) for k in INFERENCE_DEFAULTS}
        if req_speed is not None:
            params["speed"] = req_speed
        return params

    def resolve_text_splitting(text: str, language: str) -> bool:
        text_length = len(text)
        enable = should_enable_text_splitting(text_length, language)
        validate_text_splitting_configuration(text_length, language, enable)
        return enable

    def run_inference(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict, enable_split: bool):
        return model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=params["temperature"],
            length_penalty=params["length_penalty"],
            repetition_penalty=params["repetition_penalty"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            speed=params["speed"],
            enable_text_splitting=enable_split,
        )

    async def run_inference_with_timeout(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict, enable_split: bool):
        return await asyncio.wait_for(
            asyncio.to_thread(
                run_inference,
                model,
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                params,
                enable_split,
            ),
            timeout=INFERENCE_TIMEOUT_SEC,
        )

    def write_wav(wav, sr: int) -> tuple[str, Path]:
        out_name = f"{uuid.uuid4().hex}.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), wav, sr)
        return out_name, out_path

    class SynthesisRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=5000)
        voice_name: str
        language: Optional[str] = None
        speed: Optional[float] = Field(default=1.0, ge=0.1, le=2.0)
        enhancements: Optional[Dict[str, bool]] = Field(default_factory=dict)

    class ZeroShotSynthesisRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=5000)
        language: str
        speed: Optional[float] = Field(default=1.0, ge=0.1, le=2.0)
        enhancements: Optional[Dict[str, bool]] = Field(default_factory=dict)

    class SynthesisResponse(BaseModel):
        success: bool
        audio_url: str
        filename: str
        voice_used: str

    @router.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize_text(request: SynthesisRequest):
        logger.info(f"Synthesis: voice='{request.voice_name}', lang='{request.language}', text='{request.text[:60]}...'")
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        voices = [v for v in list_voices_func() if not v.is_zero_shot]
        voice = next((v for v in voices if v.name == request.voice_name), None)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found in fine-tuned voices")
         # Validate files
        ref = Path(voice.voice_path)
        mdl = Path(voice.m_path)
        voc = Path(voice.vocab_path)
        if not ref.exists():
            raise HTTPException(status_code=400, detail=f"reference.wav not found for voice '{voice.name}'")
        if not mdl.exists():
            raise HTTPException(status_code=400, detail=f"model.pth not found for voice '{voice.name}'")
        if not voc.exists():
            raise HTTPException(status_code=400, detail=f"vocab.json not found for voice '{voice.name}'")

        voice_cfg = load_voice_config(voice.config_path)
        effective_lang = validate_language(request.language, voice_cfg, supported_languages)
        logger.info(f"Effective language: {effective_lang}")
        # Load model
        model = load_xtts_func(voice.config_path, voice.m_path, voice.vocab_path)
        # Conditioning latents from voice references
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(ref)])
        # Resolve inference parameters (prioritize voice config with fallback to defaults)
        params = build_params(voice_cfg, request.speed)
        # Text splitting (spaCy)
        enable_splitting = resolve_text_splitting(request.text, effective_lang)
        try:
            out = await run_inference_with_timeout(
                model,
                request.text,
                effective_lang,
                gpt_cond_latent,
                speaker_embedding,
                params,
                enable_splitting,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="XTTS inference timed out")

        wav = out.get("wav")
        if wav is None:
            raise HTTPException(status_code=500, detail="XTTS inference returned no 'wav'")

        sample_rate = int(voice_cfg.get("audio", {}).get("output_sample_rate", 24000))
        #  optional tweaks 
        out_name, out_path = write_wav(apply_enhancements(wav, sample_rate, request.enhancements), sample_rate)

        return SynthesisResponse(
            success=True,
            audio_url=f"/files/{out_name}",
            filename=out_name,
            voice_used=voice.name,
        )

    @router.post("/synthesize/zero-shot", response_model=SynthesisResponse)
    async def synthesize_zero_shot(request: ZeroShotSynthesisRequest):
        logger.info(f"Zero-shot: lang='{request.language}', text='{request.text[:60]}...'")
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not custom_ref_path.exists():
            raise HTTPException(status_code=400, detail="No zero-shot reference uploaded")

        base_cfg = Path(zero_shot_base["config_path"])
        base_m = Path(zero_shot_base["m_path"])
        base_v = Path(zero_shot_base["vocab_path"])
        if not (base_cfg.exists() and base_m.exists() and base_v.exists()):
            raise HTTPException(status_code=500, detail="Zero-shot base model files not found")

        model = load_xtts_func(str(base_cfg), str(base_m), str(base_v))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(custom_ref_path)])

        params = build_params({}, request.speed)

        effective_lang = (request.language or "en").strip().lower()
        if effective_lang not in [l.lower() for l in supported_languages]:
            effective_lang = "en"

        enable_splitting = resolve_text_splitting(request.text, effective_lang)
        try:
            out = await run_inference_with_timeout(
                model,
                request.text,
                effective_lang,
                gpt_cond_latent,
                speaker_embedding,
                params,
                enable_splitting,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="XTTS inference timed out")

        wav = out.get("wav")
        if wav is None:
            raise HTTPException(status_code=500, detail="XTTS inference returned no 'wav'")

        # Prefer sample rate from base model config, fallback to 24000
        try:
            base_cfg_json = load_voice_config(str(base_cfg))
            sample_rate = int(base_cfg_json.get("audio", {}).get("output_sample_rate", 24000))
        except Exception:
            sample_rate = 24000
        out_name, out_path = write_wav(apply_enhancements(wav, sample_rate, request.enhancements), sample_rate)

        return SynthesisResponse(
            success=True,
            audio_url=f"/files/{out_name}",
            filename=out_name,
            voice_used="Custom",
        )

    return router


