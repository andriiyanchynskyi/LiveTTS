
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git  \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*


ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app


COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt



ARG PRELOAD_MODEL="tts_models/en/ljspeech/vits"
RUN python - << 'PY'
from TTS.api import TTS
import os
model_name = os.getenv("PRELOAD_MODEL", "tts_models/en/ljspeech/vits")
TTS(model_name)
PY


COPY app ./app


RUN mkdir -p /app/volumes/outputs

EXPOSE 8000

#  uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
