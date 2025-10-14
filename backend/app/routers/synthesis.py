import uuid
import os
import asyncio
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import soundfile as sf
import torch
import torchaudio
import numpy as np

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

    INFERENCE_TIMEOUT_SEC = float(os.getenv("INFERENCE_TIMEOUT_SEC", "360"))

    INFERENCE_DEFAULTS = {
        "temperature": 0.7,
        "length_penalty": 1.0,
        "repetition_penalty": 5.0,
        "top_k": 50,
        "top_p": 0.85,
        "speed": 1.0,
    }

    def build_params(voice_cfg: dict) -> dict:
        params = {k: get_inference_param(k, voice_cfg, INFERENCE_DEFAULTS) for k in INFERENCE_DEFAULTS}
        
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
            #speed=1.0,
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

    def _ensure_numpy(wav_input) -> np.ndarray:
        """Convert any input to numpy array."""
        if isinstance(wav_input, torch.Tensor):
            return wav_input.squeeze(0).detach().cpu().numpy()
        else:
            return wav_input


    def _apply_tempo_effect(wav_input, sr: int, speed: float):
        """Pitch-preserving tempo change. Only apply SoX on CPU when speed != 1.0.
        Keep the original device otherwise. 
        """
        # Validate speed parameter
        if not (0.5 <= speed <= 2.0):
            logger.warning(f"Invalid speed {speed}, clamping to valid range [0.5, 2.0]")
            speed = max(0.5, min(2.0, speed))
        
        #Prepare tensor [channels, num_frames]
        wav_t = torch.from_numpy(wav_input)
        
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)
        elif wav_t.dim() > 2:
            # Collapse to mono if unexpected shape
            wav_t = wav_t.view(1, -1)

        wav_t = wav_t.to(dtype=torch.float32)
        # ----- tempo change (SoX) only when needed -----
        try:
            logger.info(f"Applying SoX tempo={speed:.3f} on CPU")

            cpu_wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav_t,
                    sr,
                    [["tempo", f"{speed:.3f}"]],
            )
            cpu_wav = torch.clamp(cpu_wav, -1.0, 1.0)
        
            return cpu_wav.squeeze(0).numpy()
        except Exception as e:
            logger.error(f"SOX tempo effect failed: {e}")
            logger.warning("Returning original audio without speed modification")
            return wav_input

    def process_audio_speed_and_enhancements(
        wav, sample_rate: int, speed: Optional[float], enhancements: Dict[str, bool]
    ) -> np.ndarray:

        """Handle speed change and enhancements for any synthesis type."""
        use_sox_speed = enhancements.get("use_sox_speed", False) if enhancements else False
        target_speed = float(speed or 1.0)
        needs_sox_speed = use_sox_speed and abs(target_speed - 1.0) > 1e-3
    
        other_enhancements = {k: v for k, v in (enhancements or {}).items() if k != "use_sox_speed"}
        needs_enhancements = other_enhancements and any(other_enhancements.values())
        # Convert to tensor
        wav_final = _ensure_numpy(wav)

        if not needs_sox_speed and not needs_enhancements:
            return wav_final

        if needs_sox_speed:
            logger.info(f"SOX speed mode enabled, applying tempo effect: {target_speed}")
            wav_final = _apply_tempo_effect(wav_final, sample_rate, target_speed)

        if needs_enhancements:
            wav_final = apply_enhancements(wav_final, sample_rate, other_enhancements)

        return wav_final
        

    def save_synthesis_result(wav_process, sample_rate: int) -> tuple[str, str]:
        """Save synthesis result and return URL + filename."""
        out_name = f"{uuid.uuid4().hex}.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), wav_process, sample_rate)
        return f"/files/{out_name}", out_name

    async def run_synthesis_pipeline(
        model, text: str, language: str, gpt_cond_latent, speaker_embedding, 
        voice_cfg: dict, speed: Optional[float], enhancements: Dict[str, bool]
    ) -> tuple[str, str]:  # (audio_url, filename)
        """Shared synthesis pipeline for regular and zero-shot voices."""
        
        # Validate and normalize inputs
        target_speed = float(speed or 1.0)
        if not (0.5 <= target_speed <= 2.0):
            raise HTTPException(status_code=400, detail=f"Speed must be between 0.5 and 2.0, got {target_speed}")
        
        # Inference parameters
        params = build_params(voice_cfg)
        
        use_sox_speed = enhancements.get("use_sox_speed", False)
    
        if use_sox_speed:
            # If use SOX, XTTS should generate at speed 1.0
            params["speed"] = 1.0
            logger.info(f"SOX mode: XTTS speed=1.0, SOX will apply speed={target_speed}")
        else:
            # Use native XTTS speed (default)
            params["speed"] = target_speed
            logger.info(f"Native XTTS speed mode: speed={params['speed']}")

        # Text splitting
        enable_splitting = resolve_text_splitting(text, language)
        
        # Inference
        try:
            out = await run_inference_with_timeout(
                model, text, language, gpt_cond_latent, speaker_embedding, params, enable_splitting
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="XTTS inference timed out")
        
        wav = out.get("wav")
        if wav is None:
            raise HTTPException(status_code=500, detail="XTTS inference returned no 'wav'")
        
        # Determine sample rate
        sample_rate = int(voice_cfg.get("audio", {}).get("output_sample_rate", 24000))
        
        # Handle speed and enhancements
        wav_process = process_audio_speed_and_enhancements(wav, sample_rate, target_speed, enhancements)
        
        # Save
        return save_synthesis_result(wav_process, sample_rate)

    class SynthesisRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000)
        voice_name: str
        language: Optional[str] = None
        speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
        enhancements: Optional[Dict[str, bool]] = Field(default_factory=dict)

    class ZeroShotSynthesisRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000)
        language: str
        speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
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

        # SHARED PIPELINE
        audio_url, filename = await run_synthesis_pipeline(
            model, request.text, effective_lang, gpt_cond_latent, speaker_embedding,
            voice_cfg, request.speed, request.enhancements
        )

        return SynthesisResponse(
            success=True, audio_url=audio_url, filename=filename, voice_used=voice.name,
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

        effective_lang = (request.language or "en").strip().lower()
        if effective_lang not in [l.lower() for l in supported_languages]:
            effective_lang = "en"

        # Prefer sample rate from base model config, fallback to 24000
        try:
            base_cfg_json = load_voice_config(str(base_cfg))
        except Exception:
            base_cfg_json = {"audio": {"output_sample_rate": 24000}}

        # SHARED PIPELINE
        audio_url, filename = await run_synthesis_pipeline(
            model, request.text, effective_lang, gpt_cond_latent, speaker_embedding,
            base_cfg_json, request.speed, request.enhancements
        )

        return SynthesisResponse(
            success=True, audio_url=audio_url, filename=filename, voice_used="Custom",
        )

    return router


