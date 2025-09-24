# LiveTTS — Multilingual Text-to-Speech with XTTS v2

Production-ready FastAPI service and simple web UI for text-to-speech powered by Coqui XTTS v2. Supports fine‑tuned voice packs and temporary zero‑shot voice cloning, with optional GPU acceleration and lightweight audio post‑processing.


## Features

- **Multilingual XTTS v2**: 16 languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja
- **Fine‑tuned voices from disk**: Drop model assets under `volumes/voices/<lang>/<voice>/` with `model.pth`, `config.json`, `vocab.json`, `reference.wav`.
- **Temporary zero‑shot voice**: Upload a reference WAV via API/UI; inference runs using the bundled base at `volumes/voices/xtts_v2/`.
- **Speed control**: 0.5×–2.0× tempo (pitch‑preserving, SoX‑based).
- **Audio post‑processing (optional)**: `denoise`, `normalize`, `equalize`, `highpass`, `lowpass`.
- **GPU acceleration**: CUDA if available; optional DeepSpeed ops when supported by environment.
- **Artifact persistence**: Outputs saved to `volumes/outputs/` and served via static endpoint `/files/*`.
- **Operational introspection**: `/health`, `/gpu`, `/enhancements`, `/spacy`, plus `/healthz` for k8s‑style probes.


<img width="1314" height="1254" alt="image" src="https://github.com/user-attachments/assets/554b48a9-c658-4518-a560-04e96d831ed2" />


## Quick Start


1. **Verify Docker can access your GPU (optional):** On Windows, ensure WSL2 is enabled.
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
   ```


2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd my_tts
   ```

3. **Add voice models** to the `volumes/voices/"lang_name"/"voice_name"` directory. Requires: model.pth, config.json, vocab.json, reference.wav.

4. **Start the application**
   ```bash
   docker compose up --build
   ```

5. **Access the application** at `http://localhost:3000`. You can change ports via `docker-compose.yml`.



## API Endpoints

`GET /languages` — List supported languages

`GET /voices` — List available voice models (including temporary zero‑shot when active)

`GET /health` — Engine health and environment

`GET /gpu` — GPU status

`GET /enhancements` — Available audio post‑processing flags

`POST /synthesize/zero-shot` — Synthesize with uploaded temporary voice

### Main endpoint — POST /synthesize

Synthesize text using a fine‑tuned voice pack from `volumes/voices/*`.

Request body:
```json
{
  "text": "Hello world",
  "voice_name": "BobRoss",
  "language": "en",
  "speed": 1.0,
  "enhancements": {
    "denoise": true,
    "normalize": true,
    "equalize": false,
    "highpass": false,
    "lowpass": false
  }
}
```

Response body:
```json
{
  "success": true,
  "audio_url": "/files/6114b8eaf65045fe9eff144b1db0c39d.wav",
  "filename": "6114b8eaf65045fe9eff144b1db0c39d.wav",
  "voice_used": "BobRoss"
}
```

Notes:
- `language` is validated against the voice config; falls back to a supported default.
- Default output sample rate is taken from the voice config (fallback 24 kHz).
- Inference timeout can be controlled via `INFERENCE_TIMEOUT_SEC` (default 60s).
- You can find fine-tuned voices at [https://huggingface.co/coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2).
