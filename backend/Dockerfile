
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime


RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    sox libsox-dev libsox-fmt-all \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*


ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 


WORKDIR /app


COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download xx_ent_wiki_sm 

COPY app ./app


RUN mkdir -p /app/volumes/outputs /app/volumes/voices

EXPOSE 8000

#  uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
