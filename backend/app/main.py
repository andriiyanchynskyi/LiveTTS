import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_ROOT / "volumes" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("COQUI_TTS_MODEL", "tts_models/en/ljspeech/vits")

logger.info(f"Loading TTS model: {MODEL_NAME}")

try:
    tts = TTS(MODEL_NAME)
    logger.info("TTS model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")
    tts = None

app = FastAPI(title="Coqui TTS VITS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES_DIR = OUTPUT_DIR
app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")

class SynthesizeIn(BaseModel):
    text: str
    speaker: str | None = None
    language: str | None = None

@app.get("/health")
def health():
    if tts is None:
        raise HTTPException(500, "TTS model not loaded")
    return {"status": "ok", "model": MODEL_NAME}

@app.get("/gpu")
def gpu_info():
    try:
        import torch
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

@app.get("/voices")
def voices():
    if tts is None:
        raise HTTPException(500, "TTS model not loaded")
    
    spks = []
    try:
        spks = getattr(tts, "speakers", []) or []
    except Exception as e:
        logger.warning(f"Could not get speakers: {e}")
        pass
    if not spks:
        spks = ["default"]
    
    langs = []
    try:
        langs = getattr(tts, "languages", []) or []
    except Exception as e:
        logger.warning(f"Could not get languages: {e}")
        pass
    return {"speakers": spks, "languages": langs}

@app.post("/synthesize")
def synthesize(inp: SynthesizeIn):
    if tts is None:
        raise HTTPException(500, "TTS model not loaded")
    
    if not inp.text or not inp.text.strip():
        raise HTTPException(400, "text is empty")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    
    logger.info(f"Synthesizing text: '{inp.text[:50]}...' to {out_path}")
    logger.info(f"Output directory exists: {OUTPUT_DIR.exists()}, writable: {os.access(OUTPUT_DIR, os.W_OK)}")

    kwargs = {}
    if inp.speaker:
        kwargs["speaker"] = inp.speaker
    if inp.language:
        kwargs["language"] = inp.language

    try:
        logger.info(f"Calling TTS with kwargs: {kwargs}")
        tts.tts_to_file(text=inp.text, file_path=str(out_path), **kwargs)
        logger.info(f"TTS synthesis completed successfully to {out_path}")
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"TTS error: {e}")

    if not out_path.exists():
        logger.error(f"Output file was not created: {out_path}")
        raise HTTPException(500, "Output file was not created")

    rel_url = f"/files/{out_name}"
    return {"url": rel_url, "filename": out_name}