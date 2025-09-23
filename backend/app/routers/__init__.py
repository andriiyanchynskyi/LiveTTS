from .voices import create_router as create_voices_router
from .synthesis import create_router as create_synthesis_router
from .system import create_router as create_system_router

__all__ = [
    "create_voices_router",
    "create_synthesis_router",
    "create_system_router",
]

