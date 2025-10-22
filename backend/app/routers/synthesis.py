import uuid
import os
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Any

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
    split_text_by_sentences,
    validate_sentence_boundaries,
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
    USE_PRECISE_SPACY_SPLITTING = os.getenv("USE_PRECISE_SPACY_SPLITTING", "true").lower() in {"true", "1", "yes", "on"}

    INFERENCE_DEFAULTS = {
        "temperature": 0.7,
        "length_penalty": 1.0,
        "repetition_penalty": 5.0,
        "top_k": 50,
        "top_p": 0.85,
        "speed": 1.0,
    }

    def _validate_text_input(text: str) -> str:
        """Validate and normalize text input."""
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        return text.strip()

    def _validate_speed_parameter(speed: Optional[float]) -> float:
        """Validate speed parameter."""
        target_speed = float(speed or 1.0)
        if not (0.5 <= target_speed <= 2.0):
            raise HTTPException(status_code=400, detail=f"Speed must be between 0.5 and 2.0, got {target_speed}")
        return target_speed

    def _validate_voice_files(voice) -> None:
        """Validate that all required voice files exist."""
        ref = Path(voice.voice_path)
        mdl = Path(voice.m_path)
        voc = Path(voice.vocab_path)
        
        if not ref.exists():
            raise HTTPException(status_code=400, detail=f"reference.wav not found for voice '{voice.name}'")
        if not mdl.exists():
            raise HTTPException(status_code=400, detail=f"model.pth not found for voice '{voice.name}'")
        if not voc.exists():
            raise HTTPException(status_code=400, detail=f"vocab.json not found for voice '{voice.name}'")

    def build_params(voice_cfg: dict) -> dict:
        """Build inference parameters from voice config with defaults."""
        params = {k: get_inference_param(k, voice_cfg, INFERENCE_DEFAULTS) for k in INFERENCE_DEFAULTS}
        return params

    def should_split_text(text: str, language: str) -> bool:
        """Determine if text should be split based on length and spaCy availability."""
        text_length = len(text)
        
        # Always split if text is long enough and spaCy is available
        if text_length > 249:
            from ..utils.spacy_utils import _SPACY_AVAILABLE, _SPACY_MODELS
            if _SPACY_AVAILABLE and language in _SPACY_MODELS:
                return True
            else:
                logger.warning(f"Text is {text_length} chars but spaCy not available for '{language}'")
        
        logger.info(f"Text splitting disabled: {text_length} chars")
        return False

    def split_text_precisely(text: str, language: str, max_length: int = 250) -> List[str]:
        """
        Split text using precise spaCy sentence boundary detection.
        This replaces XTTS internal text splitting for better control.
        """
        text_length = len(text)
        
        # If text is short enough, don't split
        if text_length <= max_length:
            logger.info(f"Text length {text_length} <= {max_length}, no splitting needed")
            return [text]
        
        # Use our precise spaCy splitting
        segments = split_text_by_sentences(text, language, max_length)
        
        return segments

    def _run_model_inference(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict, enable_text_splitting: bool = False):
        """
        Centralized model inference with consistent parameter handling.
        """
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
            enable_text_splitting=enable_text_splitting,
        )

    def run_inference_single_segment(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict):
        """
        Run inference on a single text segment without internal text splitting.
        """
        return _run_model_inference(model, text, language, gpt_cond_latent, speaker_embedding, params, enable_text_splitting=False)

    def run_inference_with_precise_splitting(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict):
        """
        Run inference with precise spaCy-based text splitting.
        """
        # Split text using our precise spaCy implementation
        segments = split_text_precisely(text, language)
        
        if len(segments) == 1:
            # Single segment, run normal inference
            logger.info("Single segment, running normal inference")
            return run_inference_single_segment(model, segments[0], language, gpt_cond_latent, speaker_embedding, params)
        
        # Multiple segments, process each separately and concatenate
        all_wavs = []
        
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}: {len(segment)} chars")
            
            try:
                # Run inference on this segment
                segment_result = run_inference_single_segment(
                    model, segment, language, gpt_cond_latent, speaker_embedding, params
                )
                
                segment_wav = segment_result.get("wav")
                if segment_wav is None:
                    logger.error(f"Segment {i+1} returned no audio")
                    continue
                
                all_wavs.append(segment_wav)
                logger.info(f"Segment {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing segment {i+1}: {e}")
                # Continue with other segments
                continue
        
        if not all_wavs:
            raise RuntimeError("All segments failed to process")
        
        # Concatenate all audio segments
        if len(all_wavs) == 1:
            final_wav = all_wavs[0]
        else:
            logger.info(f"Concatenating {len(all_wavs)} audio segments")
            final_wav = np.concatenate(all_wavs, axis=0)
        
        logger.info(f"Precise splitting completed: {len(segments)} segments -> {len(final_wav)} samples")
        return {"wav": final_wav}

    def run_inference(model, text: str, language: str, gpt_cond_latent, speaker_embedding, params: dict, enable_split: bool):
        """Run inference with optional text splitting."""
        if enable_split and USE_PRECISE_SPACY_SPLITTING:
            return run_inference_with_precise_splitting(model, text, language, gpt_cond_latent, speaker_embedding, params)
        else:
            # Use XTTS internal splitting or no splitting
            return _run_model_inference(model, text, language, gpt_cond_latent, speaker_embedding, params, enable_text_splitting=enable_split)

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
        
        # Validate and normalize inputs using helper functions
        target_speed = _validate_speed_parameter(speed)
        
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
        enable_splitting = should_split_text(text, language)
        
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

    class TextSplittingTestRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000)
        language: str = "en"
        max_length: int = Field(default=250, ge=50, le=1000)

    class TextSplittingTestResponse(BaseModel):
        original_text: str
        language: str
        segments: List[str]
        segment_count: int
        total_chars: int
        validation_result: Dict[str, Any]

    @router.post("/test-text-splitting", response_model=TextSplittingTestResponse)
    async def test_text_splitting(request: TextSplittingTestRequest):
        """Test precise spaCy text splitting without synthesis."""
        logger.info(f"Testing text splitting: lang='{request.language}', text='{request.text[:60]}...'")
        
        # Get segments using our precise splitting
        segments = split_text_precisely(request.text, request.language, request.max_length)
        
        # Validate sentence boundaries
        validation_result = validate_sentence_boundaries(request.text, request.language)
        
        return TextSplittingTestResponse(
            original_text=request.text,
            language=request.language,
            segments=segments,
            segment_count=len(segments),
            total_chars=len(request.text),
            validation_result=validation_result
        )

    @router.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize_text(request: SynthesisRequest):
        logger.info(f"Synthesis: voice='{request.voice_name}', lang='{request.language}', text='{request.text[:60]}...'")
        
        # Validate text input using helper function
        text = _validate_text_input(request.text)

        voices = [v for v in list_voices_func() if not v.is_zero_shot]
        voice = next((v for v in voices if v.name == request.voice_name), None)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found in fine-tuned voices")
        
        # Validate files using helper function
        _validate_voice_files(voice)

        voice_cfg = load_voice_config(voice.config_path)
        effective_lang = validate_language(request.language, voice_cfg, supported_languages)
        logger.info(f"Effective language: {effective_lang}")
        
        # Load model
        model = load_xtts_func(voice.config_path, voice.m_path, voice.vocab_path)
        # Conditioning latents from voice references
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(voice.voice_path)])

        # SHARED PIPELINE
        audio_url, filename = await run_synthesis_pipeline(
            model, text, effective_lang, gpt_cond_latent, speaker_embedding,
            voice_cfg, request.speed, request.enhancements
        )

        return SynthesisResponse(
            success=True, audio_url=audio_url, filename=filename, voice_used=voice.name,
        )

    @router.post("/synthesize/zero-shot", response_model=SynthesisResponse)
    async def synthesize_zero_shot(request: ZeroShotSynthesisRequest):
        logger.info(f"Zero-shot: lang='{request.language}', text='{request.text[:60]}...'")
        
        # Validate text input using helper function
        text = _validate_text_input(request.text)
        
        if not custom_ref_path.exists():
            raise HTTPException(status_code=400, detail="No zero-shot reference uploaded")

        base_cfg = Path(zero_shot_base["config_path"])
        base_m = Path(zero_shot_base["m_path"])
        base_v = Path(zero_shot_base["vocab_path"])
        if not (base_cfg.exists() and base_m.exists() and base_v.exists()):
            raise HTTPException(status_code=500, detail="Zero-shot base model files not found")

        model = load_xtts_func(str(base_cfg), str(base_m), str(base_v))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(custom_ref_path)])

        # Normalize language using helper function
        effective_lang = request.language.strip().lower() if request.language else "en"
        if effective_lang not in [l.lower() for l in supported_languages]:
            effective_lang = "en"

        # Prefer sample rate from base model config, fallback to 24000
        try:
            base_cfg_json = load_voice_config(str(base_cfg))
        except Exception:
            base_cfg_json = {"audio": {"output_sample_rate": 24000}}

        # SHARED PIPELINE
        audio_url, filename = await run_synthesis_pipeline(
            model, text, effective_lang, gpt_cond_latent, speaker_embedding,
            base_cfg_json, request.speed, request.enhancements
        )

        return SynthesisResponse(
            success=True, audio_url=audio_url, filename=filename, voice_used="Custom",
        )

    return router


