# LiveTTS
Simple coqui-ai-TTS's VITS WAV-audio generator. Access via: http://localhost:5173
***
<img width="974" height="578" alt="image" src="https://github.com/user-attachments/assets/5a3ae1d8-a217-49eb-a3f4-bf5a412557e9" />

***

Enable WSL2, ensure the container initializes GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```
