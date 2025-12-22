import os
import sys
import redis
from rq import Worker, Queue, Connection
import torch
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore')

# Настройки Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = 1  # Отдельная БД для TTS

print(f"Starting TTS worker. Redis: {redis_host}:{redis_port}, DB: {redis_db}")

# Подключаемся к Redis
conn = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    decode_responses=False
)

# Проверяем соединение
try:
    conn.ping()
    print("✓ Redis connection successful")
except redis.ConnectionError as e:
    print(f"✗ Redis connection failed: {e}")
    sys.exit(1)

# Предзагружаем модели при старте воркера
print("Preloading Silero models...")
try:
    # Русская модель
    torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='aidar')
    print("✓ Russian model loaded")
    
    # Английская модель
    torch.hub.load('snakers4/silero-models', 'silero_tts', language='en', speaker='en_0')
    print("✓ English model loaded")
    
    print("All models preloaded successfully")
except Exception as e:
    print(f"Warning: Could not preload models: {e}")
    print("Models will be loaded on first request")

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(['default'])
        
        print(f"Worker started. Listening for jobs...")
        print(f"PID: {os.getpid()}")
        
        try:
            worker.work(logging_level='WARNING')
        except KeyboardInterrupt:
            print("\nWorker shutting down...")
        except Exception as e:
            print(f"Worker error: {e}")
            sys.exit(1)
