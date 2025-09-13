import os
import uuid
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import soundfile as sf

# XTTS imports
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    _XTTS_AVAILABLE = True
except Exception as _e:
    _XTTS_AVAILABLE = False
    _xtts_import_error = _e

# spaCy imports and model detection
_SPACY_AVAILABLE = False
_SPACY_MODELS: Dict[str, str] = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import spacy
except ImportError:
    logger.warning("spaCy not available...")
    
    # Language model mapping with fallback priority
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
    # Language without specified models
        "tr": ["xx_ent_wiki_sm"],
        "cs": ["xx_ent_wiki_sm"], 
        "ar": ["xx_ent_wiki_sm"],
        "hu": ["xx_ent_wiki_sm"],
        "ko": ["xx_ent_wiki_sm"]
    }
    

    for lang, models in language_models.items():
        for model in models:
            try:
                spacy.load(model)
                _SPACY_MODELS[lang] = model
                logger.info(f"Loaded spaCy model '{model}' for language '{lang}'")
                break
            except OSError:
                logger.debug(f"spaCy model '{model}' not available for language '{lang}'")
                continue
        
        # If any model is not installed, then use multilanguage fallback
        if lang not in _SPACY_MODELS:
            try:
                spacy.load("xx_ent_wiki_sm")
                _SPACY_MODELS[lang] = "xx_ent_wiki_sm"
                logger.info(f"Using multilingual model 'xx_ent_wiki_sm' as fallback for language '{lang}'")
            except OSError:
                logger.warning(f"No spaCy model available for language '{lang}'")
    
    if _SPACY_MODELS:
        logger.info(f"spaCy available with models: {_SPACY_MODELS}")
    else:
        _SPACY_AVAILABLE = False
        logger.warning("No spaCy models found, text splitting disabled")
        
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Text splitting will be disabled for long texts.")

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_ROOT / "volumes" / "outputs"
VOICES_DIR = APP_ROOT / "volumes" / "voices"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# XTTS v2 languages
SUPPORTED_LANGUAGES = [
    "en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","hu","ko","ja"
]

INFERENCE_DEFAULTS = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0
}

# ----------------- Schemas -----------------

class VoiceModel(BaseModel):
    name: str = Field(..., description="Voice name (folder)")
    language: str = Field(default="en", description="Language folder name")
    voice_path: str = Field(..., description="Path to reference.wav")
    config_path: str = Field(..., description="Path to config.json")
    vocab_path: str = Field(..., description="Path to vocab.json")
    m_path: str = Field(..., description="Path to model.pth")
    available: bool = Field(default=True)
    m_supported_languages: List[str] = Field(default_factory=list)
    fixed_language: Optional[str] = Field(default=None)

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_name: str = Field(..., description="Voice folder name")
    language: Optional[str] = Field(default=None, description="Target language code")
    speed: Optional[float] = Field(default=1.0, description="Synthesis speed (0.1-5.0)", ge=0.1, le=5.0)

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

def _normalize_path(p: str) -> str:
    if os.name == "nt":
        return p.replace("/", "\\")
    return p

def get_inference_param(param_name: str, voice_cfg: dict, defaults: dict):
    value = voice_cfg.get(param_name)
    # If value is missing, None, empty string or invalid number
    if value is None or value == '' or (isinstance(value, (int, float)) and value <= 0):
        return defaults[param_name]
    return value

def _load_voice_config(config_path: str) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as cf:
            return json.load(cf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read voice config: {e}")

def _validate_language(lang: Optional[str], voice_cfg: dict) -> str:
    supported = [l.lower() for l in voice_cfg.get("languages", SUPPORTED_LANGUAGES)]
    eff = (lang or "en").strip().lower()
    if eff not in supported:
        eff = supported[0]
    return eff

def _load_xtts_m_for_voice(config_path: str, m_path: str, vocab_path: str) -> Xtts:
    if not _XTTS_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"XTTS API not available: {_xtts_import_error}")

    try:
        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)

        # IMPORTANT: pass vocab_path to initialize the tokenizer
        # If vocab_path is missing, the model will try to look next to checkpoint_dir,
        # but we specify it explicitly for stability.
        model.load_checkpoint(
            config,
            checkpoint_path=m_path,
            vocab_path=vocab_path,
            use_deepspeed=False
        ) # [xtts.load_checkpoint]

        if torch.cuda.is_available():
            model.cuda()

        return model

    except Exception as e:
        logger.error(f"Failed to load XTTS for voice. cfg={config_path}, model={m_path}, vocab={vocab_path}, err={e}")
        raise HTTPException(status_code=500, detail=f"XTTS model load failed: {e}")

def scan_voice_directory() -> List[VoiceModel]:
    voices: List[VoiceModel] = []
    logger.info(f"Scanning voices directory: {VOICES_DIR}")

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

            if not cfg.exists():
                logger.warning(f"Missing config.json in {voice_dir}")
                continue

            if not mdl.exists():
                logger.warning(f"Missing model.pth in {voice_dir}")
                continue

            if not voc.exists():
                logger.warning(f"Missing vocab.json in {voice_dir}")
                continue

            if not ref.exists():
                logger.warning(f"Missing reference.wav in {voice_dir}")
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
                    fixed_language=fixed_lang
                )
            )

    logger.info(f"Total voices found: {len(voices)}")
    return voices

def should_enable_text_splitting(text_length: int, language: str) -> bool:
    """Check if text needs spaCy"""
    if not _SPACY_AVAILABLE:
        return False
    
   
    if language not in _SPACY_MODELS:
        return False
        
   
    return text_length > 1000

def validate_text_splitting_configuration(text_length: int, language: str, enable_splitting: bool) -> None:
    """Validate text splitting configuration and log appropriate warnings/info."""
    # Only warn if text is long but splitting is disabled
    if text_length > 1000 and not enable_splitting:
        if not _SPACY_AVAILABLE:
            logger.warning(f"Text is {text_length} characters long but spaCy is not available. "
                          "Install spaCy for automatic text splitting: pip install spacy")
        elif language not in _SPACY_MODELS:
            logger.warning(f"Text is {text_length} characters long but no spaCy model available for language '{language}'. "
                          f"Available models: {list(_SPACY_MODELS.keys())}. "
                          f"Install model: python -m spacy download en_core_web_sm")

    # Always log the current configuration
    logger.info(f"Text splitting: {'enabled' if enable_splitting else 'disabled'} "
               f"(text_length={text_length}, spacy_model={_SPACY_MODELS.get(language, 'none')})")

# ----------------- App init -----------------

app = FastAPI(
    title="Coqui TTS XTTS v2 API",
    description="Text-to-Speech API using XTTS v2 with fine-tuned per-voice checkpoints",
    version="2.2.0"
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

FILES_DIR = OUTPUT_DIR
app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")

available_voices: List[VoiceModel] = scan_voice_directory()

# ----------------- Endpoints -----------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        tts_model="xtts_v2",
        tts_type="XTTS v2",
        voices_loaded=len(available_voices)
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

@app.get("/voices", response_model=VoiceListResponse)
async def get_voices():
    global available_voices
    available_voices = scan_voice_directory()
    return VoiceListResponse(voices=available_voices, total_count=len(available_voices))

@app.get("/languages")
async def get_languages():
    global available_voices
    available_voices = scan_voice_directory()
    langs = set()
    for v in available_voices:
        try:
            cfg = _load_voice_config(v.config_path)
            langs.update([l.lower() for l in cfg.get("languages", SUPPORTED_LANGUAGES)])
        except Exception:
            langs.update([l.lower() for l in SUPPORTED_LANGUAGES])

    languages_list = sorted(langs) if langs else SUPPORTED_LANGUAGES
    return {"languages": languages_list, "total_count": len(languages_list)}

@app.get("/voices/{voice_name}", response_model=VoiceModel)
async def get_voice(voice_name: str):
    global available_voices
    available_voices = scan_voice_directory()
    voice = next((v for v in available_voices if v.name == voice_name), None)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    return voice

@app.get("/voices/{voice_name}/languages")
async def get_voice_languages(voice_name: str):
    global available_voices
    available_voices = scan_voice_directory()
    voice = next((v for v in available_voices if v.name == voice_name), None)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

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

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_text(request: SynthesisRequest):
    logger.info(f"Synthesis request: voice='{request.voice_name}', lang='{request.language}', text='{request.text[:60]}...'")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    global available_voices
    available_voices = scan_voice_directory()
    voice = next((v for v in available_voices if v.name == request.voice_name), None)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice_name}' not found. Available: {[v.name for v in available_voices]}")

    voice_cfg = _load_voice_config(voice.config_path)
    effective_lang = _validate_language(request.language, voice_cfg)
    logger.info(f"Effective language: {effective_lang}")

    # Validate required files in the voice folder
    ref = Path(voice.voice_path)
    mdl = Path(voice.m_path)
    voc = Path(voice.vocab_path)

    if not ref.exists():
        raise HTTPException(status_code=400, detail=f"reference.wav not found for voice '{voice.name}'")
    if not mdl.exists():
        raise HTTPException(status_code=400, detail=f"model.pth not found for voice '{voice.name}'")
    if not voc.exists():
        raise HTTPException(status_code=400, detail=f"vocab.json not found for voice '{voice.name}'")

    # Load the XTTS model for this specific voice
    model = _load_xtts_m_for_voice(voice.config_path, voice.m_path, voice.vocab_path)

    # Extract conditioning latents from reference.wav
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(ref)]) # mandatory step

    # Inference parameters — could be omitted, but using fallback to defaults if missing or invalid
    params = {}
    for param_name in INFERENCE_DEFAULTS:
        params[param_name] = get_inference_param(param_name, voice_cfg, INFERENCE_DEFAULTS)

    if request.speed is not None:
        params["speed"] = request.speed

    # Check text length and spaCy availability for the specific language
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

    sf.write(str(out_path), wav, sample_rate)

    return SynthesisResponse(
        success=True,
        audio_url=f"/files/{out_name}",
        filename=out_name,
        voice_used=voice.name
    )

@app.post("/voices/refresh")
async def refresh_voices():
    global available_voices
    available_voices = scan_voice_directory()
    return {"message": "Voice list refreshed", "voices_found": len(available_voices), "voices": [v.name for v in available_voices]}
