#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Silero TTS с женскими голосами
Для онлайн-школ и образовательных платформ
"""

import os
import sys
import uuid
import torch
import torchaudio
import omegaconf
import io
import tempfile
import atexit
import shutil
import threading
import time
from datetime import datetime

# Устанавливаем переменные окружения для кэша ДО всех импортов
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем директории кэша
os.makedirs('/app/cache', exist_ok=True)
os.makedirs('/app/cache/torch/hub', exist_ok=True)

# Импорты Flask
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

# Импорты Redis и очередей
import redis
from rq import Queue
from rq.job import Job

# ========== ИНИЦИАЛИЗАЦИЯ REDIS ==========
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'tts-redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1,
    socket_connect_timeout=10,
    socket_timeout=30,
    retry_on_timeout=True,
    decode_responses=False
)

# Создаем очередь задач
q = Queue(connection=redis_conn, default_timeout=600)

# ========== МОДЕЛИ SILERO ==========
MODELS = {
    'ru': 'v3_ru',
    'en': 'v3_en'
}

# ========== ИНИЦИАЛИЗАЦИЯ FLASK ==========
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
models_loading = False
models_loaded = False
startup_time = datetime.now()

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Модель для валидации входящих запросов TTS"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'  # Женский голос по умолчанию
    sample_rate: int = 24000
    put_accent: bool = True
    put_yo: bool = True
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛЕЙ ==========
def load_all_models():
    """Загрузка всех женских моделей Silero TTS при старте сервиса"""
    global tts_models, models_loading, models_loaded
    
    models_loading = True
    
    print("\n" + "=" * 60)
    print("🚀 НАЧИНАЮ ЗАГРУЗКУ МОДЕЛЕЙ SILERO TTS")
    print("=" * 60)
    
    # Проверяем доступность torch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"TorchAudio версия: {torchaudio.__version__}")
    print(f"Кэш директория: {os.environ.get('TORCH_HOME')}")
    
    # Список женских голосов для загрузки
    female_voices = [
        ('ru', 'baya'),      # Русский женский голос 1
        ('ru', 'kseniya'),   # Русский женский голос 2  
        ('ru', 'xenia'),     # Русский женский голос 3
        ('en', 'en_1'),      # Английский женский голос 1
        ('en', 'en_3'),      # Английский женский голос 2
    ]
    
    loaded_count = 0
    
    for language, speaker in female_voices:
        model_key = f"{language}_{speaker}"
        
        try:
            print(f"\n📥 Загружаю модель: {language.upper()} - '{speaker}'")
            
            # Настраиваем директорию для кэша torch hub
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Загрузка модели через torch.hub с доверием к репозиторию
            model = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=speaker,
                force_reload=False,
                verbose=False,
                trust_repo=True  # Ключевой параметр для обхода проверки безопасности
            )
            
            # Перемещаем модель на CPU (для экономии памяти)
            model.to('cpu')
            
            # Тестируем модель на коротком тексте
            try:
                test_text = "Привет" if language == 'ru' else "Hello"
                audio = model.apply_tts(
                    text=test_text,
                    speaker=speaker,
                    sample_rate=24000,
                    put_accent=True,
                    put_yo=True
                )
                test_passed = True
            except Exception as test_error:
                print(f"   ⚠️ Тест генерации не удался: {test_error}")
                test_passed = False
            
            # Сохраняем модель в кэш
            tts_models[model_key] = {
                'model': model,
                'device': 'cpu',
                'tested': test_passed,
                'loaded_at': datetime.now().isoformat()
            }
            
            loaded_count += 1
            print(f"✅ Успешно загружено: {model_key}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_key}: {str(e)[:100]}")
            
            # Детальная диагностика ошибки
            import traceback
            error_details = traceback.format_exc()
            print(f"   Детали ошибки:")
            for line in error_details.split('\n')[-5:]:
                if line.strip():
                    print(f"   {line}")
    
    models_loading = False
    models_loaded = True
    
    print("\n" + "=" * 60)
    print(f"🎯 ЗАГРУЗКА МОДЕЛЕЙ ЗАВЕРШЕНА")
    print(f"   Успешно загружено: {loaded_count} из {len(female_voices)} моделей")
    print(f"   Загруженные модели: {list(tts_models.keys())}")
    print(f"   Время старта: {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

# ========== ФУНКЦИЯ ЗАГРУЗКИ КОНКРЕТНОЙ МОДЕЛИ ==========
def load_model(language, speaker):
    """Загружает конкретную модель Silero TTS по требованию"""
    model_key = f"{language}_{speaker}"
    
    # Проверяем, есть ли модель в кэше
    if model_key not in tts_models:
        print(f"📥 Загружаю модель по требованию: {model_key}")
        
        try:
            # Настраиваем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Загружаем модель
            model = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=speaker,
                force_reload=False,
                trust_repo=True
            )
            
            # Перемещаем на CPU
            model.to('cpu')
            
            # Тестируем модель
            try:
                test_text = "Тест" if language == 'ru' else "Test"
                audio = model.apply_tts(
                    text=test_text,
                    speaker=speaker,
                    sample_rate=24000
                )
                tested = True
            except:
                tested = False
            
            # Сохраняем в кэш
            tts_models[model_key] = {
                'model': model,
                'device': 'cpu',
                'tested': tested,
                'loaded_at': datetime.now().isoformat()
            }
            
            print(f"✅ Модель загружена: {model_key}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {model_key}: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate, put_accent=True, put_yo=True):
    """Генерация аудио из текста (вызывается из фоновой задачи)"""
    try:
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Язык: {language}, Голос: {speaker}")
        print(f"   Длина текста: {len(text)} символов")
        
        start_time = time.time()
        
        # Загружаем или получаем модель из кэша
        model_info = load_model(language, speaker)
        model = model_info['model']
        
        # Разбиваем длинный текст на части (если необходимо)
        max_chars = 500
        if len(text) > max_chars:
            print(f"   Текст длинный ({len(text)} chars), разбиваю на части...")
            # Простое разбиение по предложениям
            parts = []
            current_part = ""
            for sentence in text.replace('!', '!.').replace('?', '?.').replace(';', ';.').split('.'):
                if sentence.strip():
                    if len(current_part) + len(sentence) < max_chars:
                        current_part += sentence + '.'
                    else:
                        if current_part:
                            parts.append(current_part.strip())
                        current_part = sentence + '.'
            if current_part:
                parts.append(current_part.strip())
            
            print(f"   Разбито на {len(parts)} частей")
            
            # Генерируем аудио для каждой части
            audio_parts = []
            for i, part in enumerate(parts, 1):
                print(f"   Генерация части {i}/{len(parts)}...")
                part_audio = model.apply_tts(
                    text=part,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo
                )
                audio_parts.append(part_audio)
            
            # Объединяем аудио части
            audio = torch.cat(audio_parts, dim=1)
        else:
            # Генерация для короткого текста
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )
        
        # Создаем временную директорию для аудио файлов
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Создаем временный файл
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            delete=False,
            dir=temp_dir
        )
        
        # Сохраняем аудио в файл
        torchaudio.save(
            temp_file.name, 
            audio.unsqueeze(0), 
            sample_rate,
            format='wav'
        )
        
        # Статистика генерации
        generation_time = time.time() - start_time
        audio_duration = audio.shape[1] / sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   Время генерации: {generation_time:.2f} секунд")
        print(f"   Длительность аудио: {audio_duration:.2f} секунд")
        print(f"   Размер файла: {os.path.getsize(temp_file.name) / 1024:.1f} KB")
        print(f"   Путь к файлу: {temp_file.name}")
        
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    """Главная страница TTS сервиса"""
    return render_template('index.html')

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """Основной endpoint для запроса озвучки текста"""
    try:
        # Получаем и валидируем данные
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        # Валидация через Pydantic
        req = TTSRequest(**data)
        
        # Проверка поддерживаемого языка
        if req.language not in MODELS:
            return jsonify({
                'error': f'Unsupported language: {req.language}',
                'supported_languages': list(MODELS.keys()),
                'status': 'error'
            }), 400
        
        # Проверка длины текста
        if len(req.text) == 0:
            return jsonify({
                'error': 'Text cannot be empty',
                'status': 'error'
            }), 400
        
        if len(req.text) > 5000:
            return jsonify({
                'error': f'Text too long ({len(req.text)} chars). Maximum is 5000 characters.',
                'status': 'error'
            }), 400
        
        # Логируем запрос
        print(f"\n📨 Новый TTS запрос:")
        print(f"   ID: {request.remote_addr}")
        print(f"   Язык: {req.language}, Голос: {req.speaker}")
        print(f"   Длина текста: {len(req.text)} символов")
        
        # Создаем фоновую задачу в очереди
        job = q.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate, req.put_accent, req.put_yo),
            job_timeout=300,  # 5 минут таймаут
            result_ttl=3600,  # Хранить результат 1 час
            failure_ttl=1800  # Хранить информацию о неудачных задачах 30 минут
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Task queued for processing',
            'estimated_time': '10-60 seconds depending on text length',
            'queue_position': q.get_job_position(job),
            'models_available': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except ValidationError as e:
        print(f"❌ Ошибка валидации: {e}")
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors(),
            'status': 'validation_error'
        }), 400
        
    except Exception as e:
        print(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({
                    'error': 'Job result is empty',
                    'status': 'error'
                }), 500
            
            # Проверяем, что результат - это путь к файлу
            if isinstance(result, str) and os.path.exists(result):
                try:
                    # Отправляем аудио файл
                    response = send_file(
                        result,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'tts_{job_id}.wav'
                    )
                    
                    # Удаляем файл после отправки (асинхронно)
                    @response.call_on_close
                    def cleanup_file():
                        try:
                            if os.path.exists(result):
                                os.remove(result)
                                print(f"🗑️ Удален временный файл: {result}")
                        except Exception as e:
                            print(f"⚠️ Ошибка удаления файла {result}: {e}")
                    
                    return response
                except Exception as e:
                    print(f"❌ Ошибка отправки файла: {str(e)}")
                    return jsonify({
                        'error': 'Error sending audio file',
                        'details': str(e),
                        'status': 'error'
                    }), 500
            else:
                return jsonify({
                    'error': 'Invalid job result format',
                    'status': 'error'
                }), 500
                
        elif job.is_failed:
            error_msg = str(job.exc_info) if job.exc_info else 'Unknown error'
            print(f"❌ Задача {job_id} завершилась с ошибкой: {error_msg}")
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
            
        else:
            # Задача все еще выполняется или в очереди
            position = job.get_position() if hasattr(job, 'get_position') else 'unknown'
            return jsonify({
                'status': job.get_status(),
                'position': position,
                'job_id': job_id,
                'models_loaded': list(tts_models.keys()),
                'queue_length': len(q),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        print(f"❌ Ошибка в get_status для {job_id}: {str(e)}")
        return jsonify({
            'error': f'Job not found: {str(e)}',
            'status': 'not_found'
        }), 404

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов и их статус"""
    
    # Полный список голосов (женские)
    all_voices = {
        'ru': [
            {'id': 'baya', 'name': 'Байя', 'gender': 'female', 'description': 'Чистый женский голос'},
            {'id': 'kseniya', 'name': 'Ксения', 'gender': 'female', 'description': 'Мягкий женский голос'},
            {'id': 'xenia', 'name': 'Ксения 2', 'gender': 'female', 'description': 'Альтернативный женский голос'}
        ],
        'en': [
            {'id': 'en_1', 'name': 'Emily', 'gender': 'female', 'description': 'English female voice 1'},
            {'id': 'en_3', 'name': 'Sarah', 'gender': 'female', 'description': 'English female voice 2'}
        ]
    }
    
    # Фильтруем только загруженные голоса
    loaded_voices = {}
    for lang, voices in all_voices.items():
        loaded_voices[lang] = [
            voice for voice in voices 
            if f"{lang}_{voice['id']}" in tts_models
        ]
    
    return jsonify({
        'all_voices': all_voices,
        'loaded_voices': loaded_voices,
        'total_loaded': len(tts_models),
        'models_loading': models_loading,
        'service_status': 'ready' if models_loaded else 'loading',
        'cache_size': get_cache_size('/app/cache'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса и состояния системы"""
    try:
        # Проверяем соединение с Redis
        redis_status = 'connected' if redis_conn.ping() else 'disconnected'
        
        # Собираем системную информацию
        system_info = {
            'service': 'zindaki-tts-female',
            'status': 'healthy',
            'redis': redis_status,
            'models_loaded': list(tts_models.keys()),
            'models_loading': models_loading,
            'models_loaded_count': len(tts_models),
            'queue_size': len(q),
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            'torchaudio_version': torchaudio.__version__,
            'python_version': sys.version.split()[0],
            'cache_dir': os.environ.get('TORCH_HOME'),
            'cache_size': get_cache_size('/app/cache'),
            'uptime': str(datetime.now() - startup_time),
            'startup_time': startup_time.isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(system_info), 200
        
    except Exception as e:
        print(f"❌ Ошибка health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys()),
            'torch_version': torch.__version__ if 'torch' in globals() else 'not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/load-models', methods=['POST'])
def force_load_models_endpoint():
    """Принудительная загрузка всех моделей"""
    if models_loading:
        return jsonify({
            'message': 'Models are already loading',
            'status': 'loading',
            'existing_models': list(tts_models.keys())
        }), 200
    
    # Запускаем загрузку в отдельном потоке
    thread = threading.Thread(target=load_all_models)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Model loading started in background',
        'loading': True,
        'existing_models': list(tts_models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test-voice/<language>/<speaker>', methods=['GET'])
def test_voice(language, speaker):
    """Тестирование конкретного голоса"""
    try:
        model_key = f"{language}_{speaker}"
        
        if model_key not in tts_models:
            return jsonify({
                'error': f'Voice {speaker} for language {language} not loaded',
                'status': 'not_found'
            }), 404
        
        test_text = "Привет, это тестовое сообщение." if language == 'ru' else "Hello, this is a test message."
        
        job = q.enqueue(
            generate_audio,
            args=(test_text, language, speaker, 24000, True, True),
            job_timeout=60,
            result_ttl=300
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'message': f'Test audio generation started for {speaker} ({language})',
            'test_text': test_text,
            'status': 'queued'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def get_cache_size(path):
    """Рассчитывает размер кэш директории"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return f"{total_size / (1024*1024):.2f} MB"

def cleanup_temp_files():
    """Очистка временных файлов при завершении работы"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            deleted_count = 0
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Ошибка удаления {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"🗑️ Удалено {deleted_count} временных файлов из {temp_dir}")
        except Exception as e:
            print(f"❌ Ошибка очистки временной директории: {e}")

def periodic_cache_cleanup():
    """Периодическая очистка старых временных файлов"""
    while True:
        time.sleep(3600)  # Каждый час
        try:
            temp_dir = '/app/temp_audio'
            if os.path.exists(temp_dir):
                current_time = time.time()
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        # Удаляем файлы старше 2 часов
                        if current_time - os.path.getmtime(file_path) > 7200:
                            os.unlink(file_path)
                            print(f"🗑️ Удален старый временный файл: {filename}")
        except Exception as e:
            print(f"⚠️ Ошибка периодической очистки: {e}")

# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========
if __name__ == '__main__':
    # Создаем необходимые директории
    os.makedirs('/app/temp_audio', exist_ok=True)
    os.makedirs('/app/cache', exist_ok=True)
    os.makedirs('/app/cache/torch/hub', exist_ok=True)
    
    # Регистрируем очистку при завершении
    atexit.register(cleanup_temp_files)
    
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE - FEMALE VOICES EDITION")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    print(f"🔗 Redis хост: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Запускаем периодическую очистку кэша в фоне
    cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Запускаем загрузку моделей в фоне
    print("\n⏳ Запускаю фоновую загрузку моделей TTS...")
    load_thread = threading.Thread(target=load_all_models, daemon=True)
    load_thread.start()
    
    # Даем моделям немного времени на начальную загрузку
    print("⏳ Ожидаю начальную загрузку моделей (5 секунд)...")
    time.sleep(5)
    
    # Запускаем Flask приложение
    print("\n🚀 Запускаю Flask сервер...")
    print(f"🌐 Сервер доступен по адресу: http://0.0.0.0:5000")
    print(f"🔧 Режим отладки: {'ВКЛЮЧЕН' if app.debug else 'ВЫКЛЮЧЕН'}")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )