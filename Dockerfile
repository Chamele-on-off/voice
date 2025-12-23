FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev \
    libsndfile1 ffmpeg \
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/temp_audio /app/cache

ENV TORCH_HOME=/app/cache
ENV HF_HOME=/app/cache
ENV XDG_CACHE_HOME=/app/cache

# ПРЕДЗАГРУЖАЕМ ЖЕНСКИЕ ГОЛОСА SILERO
RUN python -c "
import torch
import os
print('Preloading female TTS voices...')
os.makedirs('/app/cache/torch/hub/snakers4_silero-models_master', exist_ok=True)

# Русские женские голоса
female_voices_ru = ['baya', 'kseniya', 'xenia']
for voice in female_voices_ru:
    try:
        print(f'Loading Russian female voice: {voice}')
        torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker=voice,
            force_reload=False
        )
        print(f'✅ Russian {voice} loaded')
    except Exception as e:
        print(f'⚠️ Russian {voice} error: {e}')

# Английские женские голоса (первые 3)
female_voices_en = ['en_1', 'en_3']
for voice in female_voices_en:
    try:
        print(f'Loading English female voice: {voice}')
        torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker=voice,
            force_reload=False
        )
        print(f'✅ English {voice} loaded')
    except Exception as e:
        print(f'⚠️ English {voice} error: {e}')

print('🎀 Female voices preloaded!')
"

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]