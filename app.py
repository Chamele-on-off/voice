#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Версия 3.0
Полностью совместим с современным API Silero TTS (возвращает 2 элемента)
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
tts_models = {}
startup_time = datetime.now()

# ========== КОНФИГУРАЦИЯ SILERO TTS ==========
# Современный API возвращает только (model, example_text)
SILERO_CONFIG = {
    'ru': {
        'model': 'silero_tts',
        'language': 'ru',
        'speaker_model': 'v4_ru',  # Модель, содержащая все русские голоса
        'available_speakers': ['aidar', 'baya', 'kseniya', 'irina', 'natasha', 'ruslan'],
        'sample_rate': 24000,  # Современные модели используют 24kHz
        'example_text': 'В недрах тундры выдры в гетрах ткют в вёдра ядра кедров.'
    },
    'en': {
        'model': 'silero_tts',
        'language': 'en',
        'speaker_model': 'v3_en',
        'available_speakers': ['lj'],
        'sample_rate': 24000,
        'example_text': 'The quick brown fox jumps over the lazy dog.'
    }
}

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Валидация входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 24000  # Изменено на 24000 для совместимости
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ==========
def load_tts_model(language='ru', user_speaker='baya'):
    """
    Загружает модель Silero TTS (современный API возвращает 2 элемента)
    """
    # Формируем ключ для кэша
    model_key = f"{language}_{user_speaker}"
    
    if model_key not in tts_models:
        print(f"📥 Загружаю модель TTS: {language}/{user_speaker}")
        
        # Проверяем корректность запрошенного голоса
        if language not in SILERO_CONFIG:
            raise ValueError(f"Язык '{language}' не поддерживается")
        
        config = SILERO_CONFIG[language]
        
        if user_speaker not in config['available_speakers']:
            raise ValueError(f"Голос '{user_speaker}' не поддерживается для языка '{language}'. "
                           f"Доступные: {config['available_speakers']}")
        
        print(f"   ✅ Использую модель: {config['speaker_model']}")
        print(f"   🔊 Голос: {user_speaker}")
        print(f"   📍 torch.hub.set_dir: /app/cache/torch/hub")
        
        try:
            # Устанавливаем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Современный API возвращает (model, example_text)
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model=config['model'],
                language=config['language'],
                speaker=config['speaker_model'],
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            
            print(f"✅ Модель {config['speaker_model']} успешно загружена")
            print(f"   Пример текста: {example_text[:50]}...")
            
            # Сохраняем информацию о модели
            tts_models[model_key] = {
                'model': model,
                'example_text': example_text,
                'sample_rate': config['sample_rate'],
                'language': language,
                'speaker_model': config['speaker_model'],
                'user_speaker': user_speaker,
                'available_speakers': config['available_speakers'],
                'device': torch.device('cpu'),
                'loaded_at': datetime.now().isoformat()
            }
            
            # Перемещаем модель на CPU
            model.to(torch.device('cpu'))
            
            # Проверяем доступные методы модели
            print(f"   🔍 Проверяю методы модели...")
            if hasattr(model, 'apply_tts'):
                print(f"   ✅ Метод model.apply_tts() доступен")
            if hasattr(model, 'synthesize'):
                print(f"   ✅ Метод model.synthesize() доступен")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            print("Подробности ошибки:")
            traceback.print_exc()
            
            # Пробуем альтернативный вариант загрузки с force_reload
            print("Пробую альтернативный вариант загрузки...")
            try:
                # Очищаем кэш и пробуем снова
                cache_path = '/app/cache/torch/hub/snakers4_silero-models_master'
                if os.path.exists(cache_path):
                    print(f"🧹 Очищаю кэш модели...")
                    shutil.rmtree(cache_path)
                
                model, example_text = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model=config['model'],
                    language=config['language'],
                    speaker=config['speaker_model'],
                    force_reload=True,
                    trust_repo=True,
                    verbose=True
                )
                
                print("✅ Модель загружена с force_reload=True")
                
                # Сохраняем модель
                tts_models[model_key] = {
                    'model': model,
                    'example_text': example_text,
                    'sample_rate': config['sample_rate'],
                    'language': language,
                    'speaker_model': config['speaker_model'],
                    'user_speaker': user_speaker,
                    'available_speakers': config['available_speakers'],
                    'device': torch.device('cpu'),
                    'loaded_at': datetime.now().isoformat()
                }
                
                model.to(torch.device('cpu'))
                
            except Exception as e2:
                print(f"❌ Альтернативная загрузка тоже не удалась: {str(e2)}")
                raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """
    Генерация аудио из текста с использованием современного API Silero
    """
    try:
        start_time = time.time()
        
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Язык: {language}, Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Длина: {len(text)} символов")
        
        # Загружаем или получаем модель из кэша
        model_info = load_tts_model(language, speaker)
        
        # Получаем компоненты модели
        model = model_info['model']
        target_sample_rate = model_info['sample_rate']
        device = model_info['device']
        
        print(f"   🔊 Использую голос: {speaker}")
        print(f"   🎚️ Частота: {target_sample_rate} Hz")
        print(f"   💻 Устройство: {device}")
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Генерация аудио через современный API
        print(f"   ⚙️ Вызываю model.apply_tts()...")
        
        # Способ 1: Пробуем современный API (скорее всего сработает)
        try:
            # Вариант 1: С параметром put_accent (для русской модели)
            if language == 'ru':
                audio = model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=target_sample_rate,
                    put_accent=True,      # Расстановка ударений
                    put_yo=True,          # Использование буквы Ё
                    device=device
                )
                print(f"   ✅ Аудио сгенерировано с put_accent=True")
            else:
                # Вариант 2: Для других языков
                audio = model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=target_sample_rate,
                    device=device
                )
                print(f"   ✅ Аудио сгенерировано")
                
        except Exception as api_error:
            print(f"   ⚠️ model.apply_tts() вызвал ошибку: {api_error}")
            print(f"   🔄 Пробую альтернативный синтаксис...")
            
            # Способ 2: Альтернативный синтаксис (старые версии)
            try:
                audio = model.apply_tts(
                    texts=[text],
                    speaker=speaker,
                    sample_rate=target_sample_rate,
                    device=device
                )
                # Если возвращается список, берем первый элемент
                if isinstance(audio, list) and len(audio) > 0:
                    audio = audio[0]
                    print(f"   ✅ Аудио извлечено из списка")
                print(f"   ✅ Аудио сгенерировано (альтернативный синтаксис)")
            except Exception as alt_error:
                print(f"   ❌ Все попытки генерации не удались: {alt_error}")
                raise
        
        # Проверяем и подготавливаем аудио для сохранения
        print(f"   🔧 Подготовка аудио для сохранения...")
        
        if not hasattr(audio, 'shape'):
            raise ValueError(f"Аудио не имеет атрибута shape. Тип: {type(audio)}")
        
        print(f"   📐 Исходный shape аудио: {audio.shape}")
        
        # Приводим к правильной размерности (каналы, время)
        if audio.ndim == 1:
            # (время) -> (1, время) - один канал
            print(f"   🔄 Преобразование: 1D -> 2D (добавляем канал)")
            audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        elif audio.ndim == 2:
            # Проверяем ориентацию (каналы, время)
            if audio.shape[0] > audio.shape[1]:
                # Вероятно (время, каналы) -> транспонируем
                print(f"   🔄 Транспонируем (каналы, время)...")
                audio = audio.transpose(0, 1) if hasattr(audio, 'transpose') else audio.T
        elif audio.ndim == 3:
            # (batch, каналы, время) -> (каналы, время)
            print(f"   🔄 Убираем batch dimension...")
            audio = audio[0]
        else:
            raise ValueError(f"Неожиданная размерность аудио: {audio.ndim}")
        
        print(f"   📐 Финальный shape аудио: {audio.shape}")
        
        # Создаем временный файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            dir=temp_dir
        )
        
        # Сохраняем аудио в файл
        print(f"   💾 Сохраняю аудио в файл: {temp_file.name}")
        torchaudio.save(
            temp_file.name,
            audio,
            target_sample_rate,
            format='wav'
        )
        
        # Проверяем что файл создан
        if not os.path.exists(temp_file.name):
            raise ValueError(f"Файл не создан: {temp_file.name}")
        
        file_size = os.path.getsize(temp_file.name)
        if file_size == 0:
            raise ValueError(f"Файл пустой: {temp_file.name}")
        
        # Читаем заголовок файла для проверки формата
        with open(temp_file.name, 'rb') as f:
            header = f.read(12)
            if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                print(f"   ✅ Файл имеет корректный WAV формат")
            else:
                print(f"   ⚠️ Необычный формат файла")
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / target_sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️ Время генерации: {generation_time:.2f} секунд")
        print(f"   🕒 Длительность аудио: {audio_duration:.2f} секунд")
        print(f"   📁 Файл: {temp_file.name}")
        print(f"   📊 Размер: {file_size / 1024:.1f} KB")
        
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        traceback.print_exc()
        raise

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница с веб-интерфейсом"""
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '3.0',
            'status': 'running',
            'api': 'modern_silero_api',
            'endpoints': {
                '/': 'GET - главная страница',
                '/api/tts': 'POST - генерация аудио',
                '/api/health': 'GET - проверка здоровья',
                '/api/voices': 'GET - список голосов',
                '/api/test': 'GET - тестовый запрос'
            },
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Основной endpoint для генерации TTS
    """
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
        
        # Проверяем поддержку языка и голоса
        if req.language not in SILERO_CONFIG:
            return jsonify({
                'error': f'Language {req.language} not supported. Available: {list(SILERO_CONFIG.keys())}'
            }), 400
        
        if req.speaker not in SILERO_CONFIG[req.language]['available_speakers']:
            return jsonify({
                'error': f'Speaker {req.speaker} not supported for language {req.language}. '
                        f'Available: {SILERO_CONFIG[req.language]["available_speakers"]}'
            }), 400
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   🌐 Язык: {req.language}")
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
            'message': 'Задача добавлена в очередь обработки',
            'estimated_time': '5-30 секунд',
            'models_loaded': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat(),
            'api_version': 'silero_modern_api'
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
            position = 'unknown'
            if hasattr(job, 'get_position'):
                try:
                    position = job.get_position()
                except:
                    pass
            
            return jsonify({
                'status': status,
                'position': position,
                'models_loaded': list(tts_models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        redis_conn.ping()
        
        # Пробуем загрузить модель, если еще не загружена
        if not tts_models:
            try:
                load_tts_model('ru', 'baya')
            except Exception as e:
                print(f"⚠️ Не удалось загрузить модель при health check: {e}")
        
        model_status = {}
        for model_key, model_info in tts_models.items():
            model_status[model_key] = {
                'speaker_model': model_info.get('speaker_model', 'unknown'),
                'user_speaker': model_info.get('user_speaker', 'unknown'),
                'loaded_at': model_info.get('loaded_at', 'unknown')
            }
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '3.0',
            'api': 'silero_modern_2_element',
            'redis': 'connected',
            'models_loaded': list(tts_models.keys()),
            'models_count': len(tts_models),
            'model_details': model_status,
            'supported_languages': list(SILERO_CONFIG.keys()),
            'torch_version': torch.__version__,
            'torchaudio_version': torchaudio.__version__,
            'python_version': sys.version.split()[0],
            'uptime': str(datetime.now() - startup_time),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Список доступных голосов"""
    voices_info = {}
    
    for lang, config in SILERO_CONFIG.items():
        voices_info[lang] = []
        for speaker in config['available_speakers']:
            voice_info = {
                'id': speaker,
                'name': speaker.capitalize(),
                'language': lang,
                'sample_rate': config['sample_rate'],
                'model': config['speaker_model']
            }
            
            model_key = f"{lang}_{speaker}"
            if model_key in tts_models:
                voice_info['loaded'] = True
                voice_info['loaded_at'] = tts_models[model_key].get('loaded_at', 'unknown')
            else:
                voice_info['loaded'] = False
            
            voices_info[lang].append(voice_info)
    
    return jsonify({
        'all_voices': voices_info,
        'silero_config': SILERO_CONFIG,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint"""
    try:
        print(f"🧪 Выполняю тестовый запрос...")
        
        model_info = load_tts_model('ru', 'baya')
        
        test_text = "Привет! Это тестовое сообщение TTS сервиса."
        
        print(f"   Текст: {test_text}")
        
        model = model_info['model']
        sample_rate = model_info['sample_rate']
        device = model_info['device']
        
        # Тестируем генерацию
        try:
            audio = model.apply_tts(
                text=test_text,
                speaker='baya',
                sample_rate=sample_rate,
                put_accent=True,
                put_yo=True,
                device=device
            )
            api_type = 'modern_single_text'
        except Exception as e:
            try:
                audio = model.apply_tts(
                    texts=[test_text],
                    speaker='baya',
                    sample_rate=sample_rate,
                    device=device
                )
                if isinstance(audio, list) and len(audio) > 0:
                    audio = audio[0]
                api_type = 'modern_list_texts'
            except Exception as e2:
                api_type = 'failed'
                raise
        
        audio_shape = str(audio.shape) if hasattr(audio, 'shape') else 'no shape'
        
        print(f"   ✅ Тест успешно завершен")
        print(f"   API тип: {api_type}")
        print(f"   Формат аудио: {audio_shape}")
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'api_type': api_type,
            'audio_shape': audio_shape,
            'sample_rate': sample_rate,
            'model': model_info['speaker_model'],
            'speaker': model_info['user_speaker'],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Тестовый запрос не удался: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_details': error_details[:500],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Отладочная информация"""
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version,
        'models_loaded': list(tts_models.keys()),
        'models_detail': {k: {'speaker_model': v.get('speaker_model'), 
                             'user_speaker': v.get('user_speaker')} 
                         for k, v in tts_models.items()} if tts_models else {},
        'silero_config': SILERO_CONFIG,
        'timestamp': datetime.now().isoformat()
    })

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
    print("🎵 ZINDAKI TTS SERVICE v3.0 - Современный API Silero")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: torchaudio.__version__")
    
    print(f"\n🔧 Конфигурация Silero:")
    for lang, config in SILERO_CONFIG.items():
        print(f"   {lang.upper()}: модель={config['speaker_model']}, "
              f"голоса={config['available_speakers']}")
    
    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    print("\n⏳ Предварительная загрузка основной модели...")
    try:
        cache_path = '/app/cache/torch/hub/snakers4_silero-models_master'
        if os.path.exists(cache_path):
            print(f"🧹 Очищаю старый кэш модели...")
            shutil.rmtree(cache_path)
        
        model_info = load_tts_model('ru', 'baya')
        print(f"✅ Основная модель загружена")
        print(f"   Модель: {model_info['speaker_model']}")
        print(f"   Голос: {model_info['user_speaker']}")
        print(f"   Частота: {model_info['sample_rate']} Hz")
        
        print(f"\n🧪 Тестирую генерацию...")
        test_audio = model_info['model'].apply_tts(
            text="Тест",
            speaker='baya',
            sample_rate=model_info['sample_rate'],
            put_accent=True,
            put_yo=True,
            device=model_info['device']
        )
        print(f"✅ Тестовая генерация успешна")
        print(f"   Тип результата: {type(test_audio)}")
        if hasattr(test_audio, 'shape'):
            print(f"   Размерность: {test_audio.shape}")
            
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель при старте: {e}")
        traceback.print_exc()
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