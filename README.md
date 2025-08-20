# LiveTTS
Simple coqui-ai-TTS's VITS WAV-audio generator. Access via: http://localhost:5173
***
<img width="974" height="578" alt="image" src="https://github.com/user-attachments/assets/d496a2dd-8e27-4b24-a0f3-7740217534e5" />

***

Enable WSL2, ensure the container initializes GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```
