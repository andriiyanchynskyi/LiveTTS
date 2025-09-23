from fastapi import APIRouter
import torch

from ..utils.spacy_utils import _SPACY_AVAILABLE, _SPACY_MODELS
from ..enhancements import ENHANCEMENTS


def create_router(voices_provider, gpu_info_provider):
    router = APIRouter()

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
        return {
            "spacy_available": _SPACY_AVAILABLE,
            "loaded_models": _SPACY_MODELS,
            "total_models": len(_SPACY_MODELS)
        }

    @router.get("/enhancements")
    async def list_enhancements():
        return ENHANCEMENTS

    return router


