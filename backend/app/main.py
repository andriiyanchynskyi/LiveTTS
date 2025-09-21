import os
import uuid
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .enhancements import ENHANCEMENTS, apply_enhancements


import torch
import soundfile as sf

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Paths -----------------
APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_ROOT / "volumes" / "outputs"
VOICES_DIR = APP_ROOT / "volumes" / "voices"
CUSTOM_DIR = APP_ROOT / "volumes" / "custom"  # stores zero-shot reference + state

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_REF_PATH = CUSTOM_DIR / "reference.wav"
CUSTOM_STATE_PATH = CUSTOM_DIR / "state.json"  # persists zero-shot voice metadata

# ----------------- XTTS imports -----------------
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    _XTTS_AVAILABLE = True
except Exception as _e:
    _XTTS_AVAILABLE = False
    _xtts_import_error = _e

# ----------------- spaCy -----------------
_SPACY_AVAILABLE = False
_SPACY_MODELS: Dict[str, str] = {}

try:
    import spacy
    _SPACY_AVAILABLE = True
    # Priority list: best specific model, then multilingual fallback
    language_models = {
        "en": ["en_core_web_md", "en_core_web_sm"],
        "es": ["es_core_news_sm", "xx_ent_wiki_sm"],
        "fr": ["fr_core_news_sm", "xx_ent_wiki_sm"],
        "de": ["de_core_news_sm", "xx_ent_wiki_sm"],
        "it": ["it_core_news_sm", "xx_ent_wiki_sm"],
        "pt": ["pt_core_news_sm", "xx_ent_wiki_sm"],
        "pl": ["pl_core_news_sm", "xx_ent_wiki_sm"],
        "ru": ["ru_core_news_sm", "xx_ent_wiki_sm"],
        "nl": ["nl_core_news_sm", "xx_ent_wiki_sm"],
        "zh-cn": ["zh_core_web_sm", "xx_ent_wiki_sm"],
        "ja": ["ja_core_news_sm", "xx_ent_wiki_sm"],
        "tr": ["xx_ent_wiki_sm"],
        "cs": ["xx_ent_wiki_sm"],
        "ar": ["xx_ent_wiki_sm"],
        "hu": ["xx_ent_wiki_sm"],
        "ko": ["xx_ent_wiki_sm"]
    }
    for lang, models in language_models.items():
        for m in models:
            try:
                spacy.load(m)
                _SPACY_MODELS[lang] = m
                break
            except OSError:
                continue
    if not _SPACY_MODELS:
        _SPACY_AVAILABLE = False
        logger.warning("No spaCy models found, text splitting disabled")
    else:
        logger.info(f"spaCy models: {_SPACY_MODELS}")
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not available, text splitting disabled")

# ----------------- Constants -----------------
SUPPORTED_LANGUAGES = [
    "en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","hu","ko","ja"
]

# Inference defaults used as fallback (and as base for zero-shot)
INFERENCE_DEFAULTS = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0
}

# Base checkpoint for zero-shot mode 
ZERO_SHOT_BASE = {
    "name": "xtts_v2_base",
    # Adjust to actual model files location
    "config_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "config.json")),
    "m_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "model.pth")),
    "vocab_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "vocab.json"))
}

# ----------------- Schemas -----------------
class VoiceModel(BaseModel):
    name: str
    language: str = "en"
    voice_path: str
    config_path: str
    vocab_path: str
    m_path: str
    available: bool = True
    m_supported_languages: List[str] = Field(default_factory=list)
    fixed_language: Optional[str] = None
    # UI helpers for zero-shot virtual entry
    is_zero_shot: bool = False
    base_model: Optional[str] = None

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_name: str
    language: Optional[str] = None
    speed: Optional[float] = Field(default=1.0, ge=0.1, le=5.0)
    enhancements: Optional[Dict[str, bool]] = Field(
        default_factory=dict,
        description="Optional audio-quality enhancement flags"
    )

class ZeroShotSynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str
    speed: Optional[float] = Field(default=1.0, ge=0.1, le=5.0)
    enhancements: Optional[Dict[str, bool]] = Field(default_factory=dict)

class SynthesisResponse(BaseModel):
    success: bool
    audio_url: str
    filename: str
    voice_used: str

class VoiceListResponse(BaseModel):
    voices: List[VoiceModel]
    total_count: int

class HealthResponse(BaseModel):
    status: str
    tts_model: str
    tts_type: str
    voices_loaded: int

# ----------------- Utils -----------------
def _json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _json_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_voice_config(config_path: str) -> dict:
    try:
        return _json_load(Path(config_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read voice config: {e}")

def _validate_language(lang: Optional[str], voice_cfg: dict) -> str:
    supported = [l.lower() for l in voice_cfg.get("languages", SUPPORTED_LANGUAGES)]
    eff = (lang or "en").strip().lower()
    if eff not in supported:
        eff = supported[0]
    return eff

def _load_xtts(config_path: str, m_path: str, vocab_path: str) -> Xtts:
    if not _XTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"XTTS API not available: {_xtts_import_error}")
    try:
        cfg = XttsConfig()
        cfg.load_json(config_path)
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg, checkpoint_path=m_path, vocab_path=vocab_path, use_deepspeed=False)
        if torch.cuda.is_available():
            model.cuda()
        return model
    except Exception as e:
        logger.error(f"XTTS load failed: {e}")
        raise HTTPException(status_code=500, detail=f"XTTS model load failed: {e}")

def get_inference_param(param_name: str, voice_cfg: dict, defaults: dict):
    value = voice_cfg.get(param_name)
    if value is None or value == '' or (isinstance(value, (int, float)) and value <= 0):
        return defaults[param_name]
    return value

def should_enable_text_splitting(text_length: int, language: str) -> bool:
    if not _SPACY_AVAILABLE:
        return False
    if language not in _SPACY_MODELS:
        return False
    return text_length > 1000  # heuristic ~250 tokens

def validate_text_splitting_configuration(text_length: int, language: str, enable_splitting: bool) -> None:
    if text_length > 1000 and not enable_splitting:
        if not _SPACY_AVAILABLE:
            logger.warning(f"Text is {text_length} chars but spaCy not available.")
        elif language not in _SPACY_MODELS:
            logger.warning(f"Text is {text_length} chars but no spaCy model for '{language}'.")
    logger.info(f"Text splitting: {'enabled' if enable_splitting else 'disabled'} "
                f"(len={text_length}, spacy_model={_SPACY_MODELS.get(language, 'none')})")

def _voices_from_fs() -> List[VoiceModel]:
    """Scan VOICES_DIR to collect fine-tuned voices."""
    voices: List[VoiceModel] = []
    for lang_dir in VOICES_DIR.iterdir():
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
                # Log detailed missing parts
                if not cfg.exists(): logger.warning(f"Missing config.json in {voice_dir}")
                if not mdl.exists(): logger.warning(f"Missing model.pth in {voice_dir}")
                if not voc.exists(): logger.warning(f"Missing vocab.json in {voice_dir}")
                if not ref.exists(): logger.warning(f"Missing reference.wav in {voice_dir}")
                continue
            try:
                voice_cfg = _load_voice_config(str(cfg))
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
                    is_zero_shot=False
                )
            )
    return voices

def _zero_shot_voice_entry() -> Optional[VoiceModel]:
    """Return virtual zero-shot voice if custom reference exists and is registered."""
    if not CUSTOM_REF_PATH.exists():
        return None
    # optional metadata persisted in CUSTOM_STATE_PATH
    meta = {}
    if CUSTOM_STATE_PATH.exists():
        try:
            meta = _json_load(CUSTOM_STATE_PATH)
        except Exception:
            meta = {}
    # base model info for zero-shot
    base_cfg = ZERO_SHOT_BASE["config_path"]
    base_m = ZERO_SHOT_BASE["m_path"]
    base_v = ZERO_SHOT_BASE["vocab_path"]
    # Use 'en' as default language for UI purposes; supported languages are global SUPPORTED_LANGUAGES
    return VoiceModel(
        name=meta.get("name", "Custom"),
        language=meta.get("language", "en"),
        voice_path=str(CUSTOM_REF_PATH),
        config_path=base_cfg,
        vocab_path=base_v,
        m_path=base_m,
        available=True,
        m_supported_languages=SUPPORTED_LANGUAGES,
        fixed_language=None,
        is_zero_shot=True,
        base_model=ZERO_SHOT_BASE["name"]
    )

def _list_voices() -> List[VoiceModel]:
    """Combine FS voices with virtual zero-shot voice, if available."""
    voices = _voices_from_fs()
    zs = _zero_shot_voice_entry()
    if zs:
        # place zero-shot voice at the top of the list
        voices = [zs] + voices
    return voices



# ----------------- App init -----------------
app = FastAPI(
    title="Coqui TTS XTTS v2 API",
    description="TTS API with fine-tuned voices and temporary zero-shot integration",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# ----------------- Endpoints -----------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        tts_model="xtts_v2",
        tts_type="XTTS v2",
        voices_loaded=len(_list_voices())
    )

@app.get("/gpu")
async def gpu_info():
    try:
        info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["current_device"] = torch.cuda.current_device()
        return info
    except Exception as e:
        return {"cuda_available": False, "error": str(e)}

@app.get("/spacy")
async def spacy_info():
    return {
        "spacy_available": _SPACY_AVAILABLE,
        "loaded_models": _SPACY_MODELS,
        "total_models": len(_SPACY_MODELS)
    }

@app.get("/enhancements")
async def list_enhancements():
    """
    Return the dict the front-end uses to build toggle buttons.
    """
    return ENHANCEMENTS

@app.get("/voices", response_model=VoiceListResponse)
async def get_voices():
    voices = _list_voices()
    return VoiceListResponse(voices=voices, total_count=len(voices))

@app.post("/voices/refresh")
async def refresh_voices():
    # stateless: voices are discovered on each /voices call; here return current snapshot
    voices = _list_voices()
    return {"message": "Voice list refreshed", "voices_found": len(voices), "voices": [v.name for v in voices]}

@app.get("/languages")
async def get_languages():
    # Collect union of languages from all voices' configs; fallback to SUPPORTED_LANGUAGES
    langs = set()
    for v in _voices_from_fs():
        try:
            cfg = _load_voice_config(v.config_path)
            langs.update([l.lower() for l in cfg.get("languages", SUPPORTED_LANGUAGES)])
        except Exception:
            langs.update([l.lower() for l in SUPPORTED_LANGUAGES])
    if not langs:
        langs.update([l.lower() for l in SUPPORTED_LANGUAGES])
    languages_list = sorted(langs)
    return {"languages": languages_list, "total_count": len(languages_list)}

@app.get("/voices/{voice_name}", response_model=VoiceModel)
async def get_voice(voice_name: str):
    voices = _list_voices()
    voice = next((v for v in voices if v.name == voice_name), None)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    return voice

@app.get("/voices/{voice_name}/languages")
async def get_voice_languages(voice_name: str):
    voices = _list_voices()
    voice = next((v for v in voices if v.name == voice_name), None)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    if voice.is_zero_shot:
        # zero-shot supports global languages list
        return {
            "voice_name": voice_name,
            "supported_languages": SUPPORTED_LANGUAGES,
            "fixed_language": None,
            "voice_language": voice.language,
            "can_change_language": True
        }
    try:
        cfg = _load_voice_config(voice.config_path)
        return {
            "voice_name": voice_name,
            "supported_languages": cfg.get("languages", SUPPORTED_LANGUAGES),
            "fixed_language": cfg.get("language"),
            "voice_language": voice.language,
            "can_change_language": cfg.get("language") is None
        }
    except Exception as e:
        logger.error(f"Error reading voice config for {voice_name}: {e}")
        return {
            "voice_name": voice_name,
            "supported_languages": SUPPORTED_LANGUAGES,
            "fixed_language": None,
            "voice_language": voice.language,
            "can_change_language": True
        }

# -------- Zero-shot: upload & clear --------
@app.post("/custom-voice/upload")
async def upload_custom_voice(file: UploadFile = File(...)):
    # Basic validations
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    try:
        with open(CUSTOM_REF_PATH, "wb") as f:
            f.write(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Persist metadata for the virtual voice (optional fields)
    meta = {
        "name": "Custom",
        "language": "en"
    }
    try:
        _json_save(CUSTOM_STATE_PATH, meta)
    except Exception as e:
        logger.warning(f"Failed to save zero-shot state: {e}")

    return {"success": True, "name": meta["name"]}

@app.post("/custom-voice/clear")
async def custom_voice_clear():
    try:
        if CUSTOM_REF_PATH.exists():
            CUSTOM_REF_PATH.unlink()
        if CUSTOM_STATE_PATH.exists():
            CUSTOM_STATE_PATH.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear zero-shot voice: {e}")
    return {"success": True, "message": "Zero-shot voice cleared"}

@app.get("/custom-voice/status")
async def custom_voice_status():
    return {
        "active": CUSTOM_REF_PATH.exists(),
        "name": (_json_load(CUSTOM_STATE_PATH).get("name") if CUSTOM_STATE_PATH.exists() else None),
        "path": str(CUSTOM_REF_PATH) if CUSTOM_REF_PATH.exists() else None
    }

# ----------------- Synthesis (fine-tuned voices) -----------------
@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_text(request: SynthesisRequest):
    logger.info(f"Synthesis: voice='{request.voice_name}', lang='{request.language}', text='{request.text[:60]}...'")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    voices = _voices_from_fs()  # only real fine-tuned voices, exclude zero-shot virtual entry
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

    voice_cfg = _load_voice_config(voice.config_path)
    effective_lang = _validate_language(request.language, voice_cfg)
    logger.info(f"Effective language: {effective_lang}")

    # Load model
    model = _load_xtts(voice.config_path, voice.m_path, voice.vocab_path)

    # Conditioning latents from voice reference
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(ref)])

    # Resolve inference parameters (prioritize voice config with fallback to defaults)
    params = {}
    for p in INFERENCE_DEFAULTS:
        params[p] = get_inference_param(p, voice_cfg, INFERENCE_DEFAULTS)
    if request.speed is not None:
        params["speed"] = request.speed

    # Text splitting (spaCy)
    text_length = len(request.text)
    enable_splitting = should_enable_text_splitting(text_length, effective_lang)
    validate_text_splitting_configuration(text_length, effective_lang, enable_splitting)

    out = model.inference(
        request.text,
        effective_lang,
        gpt_cond_latent,
        speaker_embedding,
        temperature=params["temperature"],
        length_penalty=params["length_penalty"],
        repetition_penalty=params["repetition_penalty"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        speed=params["speed"],
        enable_text_splitting=enable_splitting
    )

    wav = out.get("wav")
    if wav is None:
        raise HTTPException(status_code=500, detail="XTTS inference returned no 'wav'")

    sample_rate = int(voice_cfg.get("audio", {}).get("output_sample_rate", 24000))
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name

    # --- optional tweaks ---------------------------------
    wav = apply_enhancements( wav, sample_rate, request.enhancements)
   
    sf.write(str(out_path), wav, sample_rate)

    return SynthesisResponse(
        success=True,
        audio_url=f"/files/{out_name}",
        filename=out_name,
        voice_used=voice.name
    )

# ----------------- Synthesis (zero-shot) -----------------
@app.post("/synthesize/zero-shot", response_model=SynthesisResponse)
async def synthesize_zero_shot(request: ZeroShotSynthesisRequest):
    logger.info(f"Zero-shot: lang='{request.language}', text='{request.text[:60]}...'")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if not CUSTOM_REF_PATH.exists():
        raise HTTPException(status_code=400, detail="No zero-shot reference uploaded")

    # Validate base model files
    base_cfg = Path(ZERO_SHOT_BASE["config_path"])
    base_m = Path(ZERO_SHOT_BASE["m_path"])
    base_v = Path(ZERO_SHOT_BASE["vocab_path"])
    if not (base_cfg.exists() and base_m.exists() and base_v.exists()):
        raise HTTPException(status_code=500, detail="Zero-shot base model files not found")

    # Load base model
    model = _load_xtts(str(base_cfg), str(base_m), str(base_v))

    # Conditioning latents from custom reference
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(CUSTOM_REF_PATH)])

    # Use universal defaults as base params for zero-shot
    params = dict(INFERENCE_DEFAULTS)
    if request.speed is not None:
        params["speed"] = request.speed

    effective_lang = (request.language or "en").strip().lower()
    if effective_lang not in [l.lower() for l in SUPPORTED_LANGUAGES]:
        effective_lang = "en"

    # Text splitting (spaCy)
    text_length = len(request.text)
    enable_splitting = should_enable_text_splitting(text_length, effective_lang)
    validate_text_splitting_configuration(text_length, effective_lang, enable_splitting)

    out = model.inference(
        request.text,
        effective_lang,
        gpt_cond_latent,
        speaker_embedding,
        temperature=params["temperature"],
        length_penalty=params["length_penalty"],
        repetition_penalty=params["repetition_penalty"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        speed=params["speed"],
        enable_text_splitting=enable_splitting
    )

    wav = out.get("wav")
    if wav is None:
        raise HTTPException(status_code=500, detail="XTTS inference returned no 'wav'")

    # In zero-shot, use a safe default sample rate (or configure in ZERO_SHOT_BASE meta if available)
    sample_rate = 24000
    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    
    # --- optional tweaks ---------------------------------
    wav = apply_enhancements( wav, sample_rate, request.enhancements)
   
    
    sf.write(str(out_path), wav, sample_rate)

    return SynthesisResponse(
        success=True,
        audio_url=f"/files/{out_name}",
        filename=out_name,
        voice_used="Custom"
    )
 
 