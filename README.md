# LiveTTS - Multilingual Text-to-Speech with XTTS v2

A web application for text-to-speech synthesis using Coqui TTS XTTS v2 model with support for multiple languages and custom voice models.

## Features

**Multilingual Support**: Supports 17 languages
**Custom Voice Models**: Load your XTTS models into /Voices folder
**Real-time Synthesis**: Generate high-quality speech from text input
**GPU Acceleration**: Optional CUDA support for faster processing


## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd my_tts
   ```

2. **Add voice models** to the `volumes/voices/"lang_name"/"voice_name"` directory. Requires: model.pth; config.json; vocal.json; reference.wav.

3. **Start the application**
   ```bash
   docker compose up --build
   ```

4. **Access the application** at `http://localhost:3000`



Enable WSL2, ensure the container initializes GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```


## API Endpoints

`GET /languages` - Get list of supported languages

`GET /voices` - Get list of available voice models

`POST /synthesize` - Synthesize text to speech

`GET /health` - Sound engine's health check

`GET /gpu` - GPU status information for an additional CUDA processing


