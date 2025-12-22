FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p temp_audio && \
    mkdir -p templates && \
    chmod 755 temp_audio

# Предзагружаем модели Silero (опционально, можно загружать при первом запросе)
RUN python -c "
import torch
print('Downloading Silero models...')
torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='aidar')
torch.hub.load('snakers4/silero-models', 'silero_tts', language='en', speaker='en_0')
print('Models downloaded successfully')
"

# Порт для Flask приложения
EXPOSE 5000

# Команда запуска
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
