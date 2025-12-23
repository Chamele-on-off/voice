#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Минимальная рабочая версия
Для старых версий Silero TTS
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import redis
from rq import Queue
from rq.job import Job
import threading
import atexit
import traceback

# ========== НАСТРОЙКА ОКРУЖЕНИЯ ==========
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем необходимые директории
os.makedirs('/app/cache/torch/hub', exist_ok=True)
os.makedirs('/app/temp_audio', exist_ok=True)
os.makedirs('/app/templates', exist_ok=True)

# ========== ИНИЦИАЛИЗАЦИЯ REDIS ==========
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'tts-redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1,
    socket_connect_timeout=10,
    socket_timeout=30,
    retry_on_timeout=True
)

# Очередь задач
queue = Queue(connection=redis_conn, default_timeout=600)

# ========== НАСТРОЙКА FLASK ==========
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_model = None  # Храним одну модель
tts_example_text = ""
startup_time = datetime.now()
model_loaded = False

# ========== КОНФИГУРАЦИЯ SILERO TTS ==========
# Простейшая конфигурация для старых версий
SILERO_CONFIG = {
    'ru': {
        'speakers': ['aidar', 'baya', 'kseniya', 'irina', 'natasha', 'ruslan'],
        'sample_rate': 16000,  # Старые версии используют 16kHz
        'model_name': 'v3_ru'  # Пробуем старую версию
    }
}

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Валидация входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 16000  # 16kHz для старых версий
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ==========
def load_tts_model():
    """Загружает модель Silero TTS (минимальная версия для старых API)"""
    global tts_model, tts_example_text, model_loaded
    
    if tts_model is None:
        print(f"📥 Загружаю модель Silero TTS...")
        
        try:
            # Устанавливаем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Пробуем разные версии моделей
            model_versions = ['v3_ru', 'v3_1_ru', 'ru_v3', 'v4_ru']
            
            for model_version in model_versions:
                try:
                    print(f"   Пробую версию: {model_version}")
                    tts_model, tts_example_text = torch.hub.load(
                        repo_or_dir='snakers4/silero-models',
                        model='silero_tts',
                        language='ru',
                        speaker=model_version,
                        force_reload=False,
                        trust_repo=True,
                        verbose=False
                    )
                    print(f"✅ Модель {model_version} успешно загружена")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"   ❌ {model_version} не сработала: {str(e)[:80]}")
                    continue
            
            if tts_model is None:
                raise ValueError("Не удалось загрузить ни одну версию модели")
                
            # Тестируем простейший вызов
            print(f"   🔍 Тестирую простейший вызов...")
            try:
                # Самая простая версия вызова
                audio = tts_model.apply_tts(
                    texts=["Тест"],
                    speaker='baya',
                    sample_rate=16000
                )
                print(f"   ✅ API работает с texts=[], speaker=, sample_rate=")
            except Exception as e:
                print(f"   ❌ Простейший вызов не работает: {e}")
                raise
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            traceback.print_exc()
            raise
    
    return tts_model, tts_example_text

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """Генерация аудио из текста (минимальная версия для старых API)"""
    try:
        start_time = time.time()
        
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Загружаем модель
        model, _ = load_tts_model()
        
        # Простейший вызов без лишних параметров
        print(f"   ⚙️ Вызываю model.apply_tts()...")
        
        try:
            # Вариант 1: Минимальный вызов
            audio = model.apply_tts(
                texts=[text],      # Всегда список
                speaker=speaker,
                sample_rate=16000  # Фиксированная частота
            )
            print(f"   ✅ Использован минимальный вызов")
        except Exception as e1:
            print(f"   ⚠️ Первый вариант не сработал: {e1}")
            # Вариант 2: Без sample_rate
            try:
                audio = model.apply_tts(
                    texts=[text],
                    speaker=speaker
                )
                print(f"   ✅ Использован вызов без sample_rate")
            except Exception as e2:
                print(f"   ❌ Все варианты не сработали")
                raise
        
        # Извлекаем аудио если это список
        if isinstance(audio, list) and len(audio) > 0:
            audio = audio[0]
            print(f"   📊 Извлечен аудио из списка")
        
        # Проверяем аудио
        if not hasattr(audio, 'shape'):
            raise ValueError(f"Аудио не имеет атрибута shape. Тип: {type(audio)}")
        
        print(f"   📐 Shape аудио: {audio.shape}")
        
        # Приводим к правильной размерности
        if audio.ndim == 1:
            audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.transpose(0, 1) if hasattr(audio, 'transpose') else audio.T
        
        print(f"   📐 Финальный shape: {audio.shape}")
        
        # Сохраняем в файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=temp_dir
        )
        
        print(f"   💾 Сохраняю аудио в файл: {temp_file.name}")
        torchaudio.save(
            temp_file.name,
            audio,
            16000,  # Фиксированная частота
            format='wav'
        )
        
        # Проверяем файл
        if not os.path.exists(temp_file.name):
            raise ValueError(f"Файл не создан")
        
        file_size = os.path.getsize(temp_file.name)
        if file_size == 0:
            raise ValueError(f"Файл пустой")
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / 16000
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️ Время: {generation_time:.2f} сек")
        print(f"   🕒 Длительность: {audio_duration:.2f} сек")
        print(f"   📊 Размер: {file_size / 1024:.1f} KB")
        
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        traceback.print_exc()
        raise

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница"""
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '5.0',
            'status': 'running',
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """Основной endpoint для генерации TTS"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        req = TTSRequest(**data)
        
        # Проверка длины текста
        if len(req.text) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(req.text) > 5000:
            return jsonify({
                'error': f'Text too long ({len(req.text)} chars). Max is 5000.'
            }), 400
        
        # Проверяем голос
        if req.speaker not in SILERO_CONFIG['ru']['speakers']:
            return jsonify({
                'error': f'Speaker {req.speaker} not supported. Available: {SILERO_CONFIG["ru"]["speakers"]}'
            }), 400
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   🗣️ Голос: {req.speaker}")
        print(f"   📝 Длина текста: {len(req.text)} символов")
        
        # Создаем фоновую задачу
        job = queue.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate),
            job_timeout=300,
            result_ttl=3600,
            failure_ttl=1800
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача добавлена в очередь',
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except ValidationError as e:
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors()
        }), 400
        
    except Exception as e:
        print(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Проверка статуса задачи"""
    try:
        print(f"📋 Проверка статуса задачи: {job_id}")
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            print(f"✅ Задача завершена: {job_id}")
            result = job.result
            
            if not result or not os.path.exists(result):
                return jsonify({'error': 'Audio file not found'}), 500
            
            file_size = os.path.getsize(result)
            if file_size == 0:
                return jsonify({'error': 'Audio file is empty'}), 500
            
            print(f"📤 Отправляю аудио файл: {result} ({file_size} bytes)")
            
            response = send_file(
                result,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'tts_{job_id}.wav'
            )
            
            @response.call_on_close
            def cleanup():
                try:
                    if os.path.exists(result):
                        os.remove(result)
                        print(f"🗑️ Удален временный файл: {result}")
                except Exception as e:
                    print(f"⚠️ Ошибка удаления файла: {e}")
            
            return response
            
        elif job.is_failed:
            error_msg = str(job.exc_info) if job.exc_info else 'Unknown error'
            print(f"❌ Задача завершилась с ошибкой: {error_msg}")
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
            
        else:
            status = job.get_status()
            return jsonify({
                'status': status,
                'model_loaded': model_loaded,
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        redis_conn.ping()
        
        # Пробуем загрузить модель
        if not model_loaded:
            try:
                load_tts_model()
            except Exception as e:
                print(f"⚠️ Не удалось загрузить модель: {e}")
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '5.0',
            'model_loaded': model_loaded,
            'torch_version': torch.__version__,
            'python_version': sys.version.split()[0],
            'uptime': str(datetime.now() - startup_time),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Список доступных голосов"""
    voices = []
    
    for speaker in SILERO_CONFIG['ru']['speakers']:
        voices.append({
            'id': speaker,
            'name': speaker.capitalize(),
            'language': 'ru',
            'sample_rate': 16000,
            'loaded': model_loaded,
            'status': '✅ Загружен' if model_loaded else '❌ Не загружен'
        })
    
    return jsonify({
        'all_voices': {'ru': voices},
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint"""
    try:
        print(f"🧪 Выполняю тестовый запрос...")
        
        # Загружаем модель
        model, example_text = load_tts_model()
        
        test_text = "Привет! Тест."
        print(f"   Текст: {test_text}")
        
        # Минимальный тестовый вызов
        audio = model.apply_tts(
            texts=[test_text],
            speaker='baya',
            sample_rate=16000
        )
        
        # Извлекаем аудио если нужно
        if isinstance(audio, list) and len(audio) > 0:
            audio = audio[0]
        
        audio_shape = str(audio.shape) if hasattr(audio, 'shape') else 'no shape'
        
        print(f"   ✅ Тест успешно завершен")
        print(f"   Формат аудио: {audio_shape}")
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'audio_shape': audio_shape,
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Тестовый запрос не удался: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        }), 500

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def cleanup_temp_files():
    """Очистка временных файлов"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            count = 0
            current_time = time.time()
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 3600:
                            os.remove(file_path)
                            count += 1
                    except:
                        pass
            if count > 0:
                print(f"🗑️ Удалено {count} старых временных файлов")
        except Exception as e:
            print(f"⚠️ Ошибка очистки временных файлов: {e}")

def periodic_cleanup():
    """Периодическая очистка"""
    while True:
        time.sleep(3600)
        cleanup_temp_files()

atexit.register(cleanup_temp_files)

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE v5.0 - Минимальная версия")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    
    # Запускаем очистку
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    print("\n⏳ Пробую загрузить модель...")
    try:
        # Очищаем старый кэш
        cache_path = '/app/cache/torch/hub/snakers4_silero-models_master'
        if os.path.exists(cache_path):
            print(f"🧹 Очищаю старый кэш модели...")
            shutil.rmtree(cache_path)
        
        # Загружаем модель
        load_tts_model()
        print(f"✅ Модель загружена: {model_loaded}")
        
        # Тестируем
        if model_loaded:
            print(f"\n🧪 Тестирую генерацию...")
            model, _ = load_tts_model()
            audio = model.apply_tts(
                texts=["Тест"],
                speaker='baya',
                sample_rate=16000
            )
            print(f"✅ Тестовая генерация успешна")
            if isinstance(audio, list):
                print(f"   Результат список из {len(audio)} элементов")
            else:
                print(f"   Тип результата: {type(audio)}")
            
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель при старте: {e}")
        print("   Модель будет загружена при первом запросе")
    
    print("\n🚀 Запуск Flask сервера...")
    print(f"🌐 Доступен по адресу: http://0.0.0.0:5000")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )