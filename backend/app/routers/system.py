from fastapi import APIRouter
from pydantic import BaseModel, Field
import torch

from ..utils.spacy_utils import (
    _SPACY_AVAILABLE, 
    _SPACY_MODELS, 
    validate_sentence_boundaries,
    split_text_by_sentences
)
from ..enhancements import ENHANCEMENTS


def create_router(voices_provider, gpu_info_provider):
    router = APIRouter()

    class TextValidationRequest(BaseModel):
        text: str
        language: str = Field(default="en", description="Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zn-cn, hu, ko, ja)")

    @router.get("/health")
    async def health_check():
        voices = voices_provider()
        return {
            "status": "healthy",
            "tts_model": "xtts_v2",
            "tts_type": "XTTS v2",
            "voices_loaded": len(voices),
            "voices_zero_shot": sum(1 for v in voices if getattr(v, 'is_zero_shot', False)),
            "spacy_available": _SPACY_AVAILABLE,
            "spacy_models": _SPACY_MODELS,
            "gpu": {
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count(),
                "devices": [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                    }
                    for i in range(torch.cuda.device_count())
                ] if torch.cuda.is_available() else []
            }
        }

    @router.get("/gpu")
    async def gpu_info():
        return gpu_info_provider()

    @router.get("/spacy")
    async def spacy_info():
        import os
        use_precise_splitting = os.getenv("USE_PRECISE_SPACY_SPLITTING", "true").lower() in {"true", "1", "yes", "on"}
        
        return {
            "spacy_available": _SPACY_AVAILABLE,
            "loaded_models": _SPACY_MODELS,
            "total_models": len(_SPACY_MODELS),
            "precise_splitting_enabled": use_precise_splitting,
            "splitting_method": "precise_spacy" if use_precise_splitting else "xtts_internal"
        }

    @router.post("/validate-text-splitting")
    async def validate_text_splitting(request: TextValidationRequest):
        """Validate spaCy sentence boundary detection for debugging."""
        result = validate_sentence_boundaries(request.text, request.language)
        return result

    @router.post("/split-text")
    async def split_text(request: TextValidationRequest):
        """Split text using spaCy sentence boundaries for debugging."""
        segments = split_text_by_sentences(request.text, request.language)
        return {
            "original_text": request.text,
            "language": request.language,
            "segments": segments,
            "segment_count": len(segments),
            "total_chars": len(request.text)
        }

    @router.get("/enhancements")
    async def list_enhancements():
        return ENHANCEMENTS

    return router


