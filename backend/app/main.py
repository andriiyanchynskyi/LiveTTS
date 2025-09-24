import os
import logging
from pathlib import Path


from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from urllib.parse import urlparse


from .routers import create_voices_router,  create_synthesis_router, create_system_router

from .repositories.voice_repo import list_voices


import torch

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


# ----------------- Constants -----------------
SUPPORTED_LANGUAGES = [
    "en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","hu","ko","ja"
]


# Base checkpoint for zero-shot mode 
ZERO_SHOT_BASE = {
    "name": "xtts_v2_base",
    # Adjust to actual model files location
    "config_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "config.json")),
    "m_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "model.pth")),
    "vocab_path": str((APP_ROOT / "volumes" / "voices" / "xtts_v2" / "vocab.json"))
}

# ----------------- Utils -----------------

def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1","true","yes","on"}

def _load_xtts(config_path: str, m_path: str, vocab_path: str) -> Xtts:
    if not _XTTS_AVAILABLE:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"XTTS API not available: {_xtts_import_error}")
    try:
        cfg = XttsConfig()
        cfg.load_json(config_path)
     
        model = Xtts.init_from_config(cfg)

        want_ds = _env_truthy("USE_DEEPSPEED")
        cuda_ok = torch.cuda.is_available()

        if want_ds and "CUDA_HOME" not in os.environ:
            logger.warning("No CUDA_HOME; force DS_BUILD_OPS=0 in the docker's environment")

        try:
            model.load_checkpoint(cfg, checkpoint_path=m_path, vocab_path=vocab_path,
                                  use_deepspeed= want_ds and cuda_ok)
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["cuda_home", "deepspeed", "nvcc", "build", "CUDA_HOME"]):
                logger.error("DeepSpeed ops build failed, reloading without DS")
                model.load_checkpoint(cfg, checkpoint_path=m_path, vocab_path=vocab_path,
                                      use_deepspeed=False)
            else:
                raise

        if torch.cuda.is_available():
            model.cuda()
        return model 
    except Exception as e:
        logger.error(f"XTTS load failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"XTTS model load failed: {e}")


# ----------------- App init -----------------
app = FastAPI(
    title="Coqui TTS XTTS v2 API",
    description="TTS API with fine-tuned voices and temporary zero-shot integration",
    version="2.5.0"
)

# ----------------- CORS -----------------
_cors_env = os.getenv("CORS_ORIGINS", "").strip()

def _expand_origins_with_ports(origins, ports=(5173, 3000)):
    expanded = set()
    for origin in origins:
        if not origin:
            continue
        if origin == "*":
            expanded.add("*")
            continue
        parsed = urlparse(origin if "://" in origin else f"http://{origin}")
        scheme = parsed.scheme or "http"
        host = parsed.hostname or ""
        port = parsed.port

        # Always include the normalized origin itself
        normalized = f"{scheme}://{host}{f':{port}' if port else ''}"
        if normalized and normalized not in expanded:
            expanded.add(normalized)

        # For localhost/127.0.0.1 without explicit port, add common dev ports
        if port is None and host in {"localhost", "127.0.0.1"}:
            for p in ports:
                candidate = f"{scheme}://{host}:{p}"
                if candidate not in expanded:
                    expanded.add(candidate)
    return sorted(expanded)

if _cors_env:
    _raw_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    # Sensible dev defaults when CORS_ORIGINS is not provided
    _raw_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

ALLOW_ORIGINS = _expand_origins_with_ports(_raw_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# ----------------- Global error handlers -----------------
def _http_exception_to_json(exception: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exception.status_code, content={"error": exception.detail})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return _http_exception_to_json(exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Validation error", "details": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ----------------- Lightweight health endpoint -----------------
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok" if _XTTS_AVAILABLE else "degraded",
        "xtts_available": _XTTS_AVAILABLE,
        **({"xtts_error": str(_xtts_import_error)} if not _XTTS_AVAILABLE else {}),
    }

# ----------------- Router wiring -----------------
app.include_router(
    create_system_router(
        voices_provider=lambda: list_voices(VOICES_DIR, CUSTOM_REF_PATH, CUSTOM_STATE_PATH, ZERO_SHOT_BASE, SUPPORTED_LANGUAGES),
        gpu_info_provider=lambda: {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
            **({"device_name": torch.cuda.get_device_name(0), "current_device": torch.cuda.current_device()} if torch.cuda.is_available() else {})
        },
    )
)

app.include_router(
    create_voices_router(
        voices_dir=VOICES_DIR,
        custom_ref_path=CUSTOM_REF_PATH,
        custom_state_path=CUSTOM_STATE_PATH,
        zero_shot_base=ZERO_SHOT_BASE,
        supported_languages=SUPPORTED_LANGUAGES,
    )
)

app.include_router(
    create_synthesis_router(
        list_voices_func=lambda: list_voices(VOICES_DIR, CUSTOM_REF_PATH, CUSTOM_STATE_PATH, ZERO_SHOT_BASE, SUPPORTED_LANGUAGES),
        voices_dir=VOICES_DIR,
        output_dir=OUTPUT_DIR,
        custom_ref_path=CUSTOM_REF_PATH,
        zero_shot_base=ZERO_SHOT_BASE,
        supported_languages=SUPPORTED_LANGUAGES,
        load_xtts_func=_load_xtts,
        logger=logger,
    )
)
 
 