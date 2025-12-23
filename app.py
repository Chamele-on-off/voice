#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Исправленная версия для Silero v4/v5
Исправлена ошибка загрузки модели с новым API Silero
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
import json

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

# ========== КОРРЕКТНЫЕ ПАРАМЕТРЫ ДЛЯ СОВРЕМЕННОГО SILERO ==========
# Для v4_ru модели используем эти параметры
SILERO_CONFIG = {
    'ru': {
        'model': 'silero_tts',
        'language': 'ru',
        'speaker_model': 'v4_ru',  # Модель, которая содержит все русские голоса
        'available_speakers': ['aidar', 'baya', 'kseniya', 'irina', 'natasha', 'ruslan'],
        'sample_rate': 16000,
        'example_text': 'В недрах тундры выдры в гетрах ткют в вёдра ядра кедров.'
    },
    'en': {
        'model': 'silero_tts',
        'language': 'en',
        'speaker_model': 'v3_en',  # Английская модель
        'available_speakers': ['lj'],
        'sample_rate': 16000,
        'example_text': 'The quick brown fox jumps over the lazy dog.'
    }
}

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Валидация входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 16000
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ==========
def load_tts_model(language='ru', user_speaker='baya'):
    """
    Загружает модель Silero TTS (современная версия API)
    Использует v4_ru модель, которая содержит все русские голоса
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
            
            # ВНИМАНИЕ: Для современных версий Silero:
            # 1. Загружаем модель v4_ru (содержит все русские голоса)
            # 2. speaker параметр при загрузке модели - это имя МОДЕЛИ (v4_ru), а не конкретного голоса
            # 3. Конкретный голос (baya, aidar) указываем позже в speakers параметре apply_tts
            model, symbols, sample_rate, _, apply_tts = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model=config['model'],
                language=config['language'],
                speaker=config['speaker_model'],  # Имя МОДЕЛИ, а не голоса!
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            
            print(f"✅ Модель {config['speaker_model']} успешно загружена")
            print(f"   Поддерживаемые голоса: {config['available_speakers']}")
            
            # Сохраняем все компоненты
            tts_models[model_key] = {
                'model': model,
                'symbols': symbols,
                'sample_rate': sample_rate,
                'apply_tts': apply_tts,
                'language': language,
                'speaker_model': config['speaker_model'],  # Модель (v4_ru)
                'user_speaker': user_speaker,  # Конкретный голос (baya)
                'available_speakers': config['available_speakers'],
                'device': torch.device('cpu'),
                'loaded_at': datetime.now().isoformat()
            }
            
            # Перемещаем модель на CPU
            tts_models[model_key]['model'].to(tts_models[model_key]['device'])
            
            print(f"   🎵 Sample rate: {sample_rate} Hz")
            
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
                
                model, symbols, sample_rate, _, apply_tts = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model=config['model'],
                    language=config['language'],
                    speaker=config['speaker_model'],
                    force_reload=True,  # Принудительная перезагрузка
                    trust_repo=True,
                    verbose=True
                )
                
                print("✅ Модель загружена с force_reload=True")
                
                # Сохраняем модель
                tts_models[model_key] = {
                    'model': model,
                    'symbols': symbols,
                    'sample_rate': sample_rate,
                    'apply_tts': apply_tts,
                    'language': language,
                    'speaker_model': config['speaker_model'],
                    'user_speaker': user_speaker,
                    'available_speakers': config['available_speakers'],
                    'device': torch.device('cpu'),
                    'loaded_at': datetime.now().isoformat()
                }
                
                tts_models[model_key]['model'].to(tts_models[model_key]['device'])
                
            except Exception as e2:
                print(f"❌ Альтернативная загрузка тоже не удалась: {str(e2)}")
                raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """
    Генерация аудио из текста
    Ключевое изменение: передаем speakers=[speaker] в apply_tts
    """
    try:
        start_time = time.time()
        
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Язык: {language}, Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Длина: {len(text)} символов")
        
        # Загружаем или получаем модель из кэша
        model_info = load_tts_model(language, speaker)
        
        # Проверяем, что модель загрузилась корректно
        if not model_info['apply_tts']:
            raise ValueError("Модель не содержит функцию apply_tts")
        
        # Получаем компоненты модели
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']
        
        print(f"   🔊 Использую голос: {speaker}")
        print(f"   🎚️  Частота: {target_sample_rate} Hz")
        print(f"   💻 Устройство: {device}")
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Генерация аудио с указанием speakers
        print(f"   ⚙️  Вызываю apply_tts с speakers=[{speaker}]...")
        
        # Вариант 1: Современный API с speakers параметром
        try:
            audio_result = apply_tts_func(
                texts=[text],           # Список текстов
                model=model,            # Модель TTS
                sample_rate=target_sample_rate,  # Частота дискретизации
                symbols=symbols,        # Алфавит/символы
                device=device,          # Устройство (CPU)
                speakers=[speaker]      # КЛЮЧЕВОЙ ПАРАМЕТР: конкретный голос
            )
            print(f"   ✅ Использован API с speakers параметром")
        except TypeError as e:
            # Вариант 2: Старый API (если speakers не поддерживается)
            print(f"   ⚠️  speakers не поддерживается, пробую старый API...")
            audio_result = apply_tts_func(
                texts=[text],
                model=model,
                sample_rate=target_sample_rate,
                symbols=symbols,
                device=device
            )
            print(f"   ✅ Использован старый API (без speakers)")
        
        # ДИАГНОСТИКА: выводим тип результата
        print(f"   📊 Тип результата apply_tts: {type(audio_result)}")
        if isinstance(audio_result, list):
            print(f"   📊 Длина списка: {len(audio_result)}")
            if len(audio_result) > 0:
                print(f"   📊 Тип первого элемента: {type(audio_result[0])}")
                if hasattr(audio_result[0], 'shape'):
                    print(f"   📊 Shape первого элемента: {audio_result[0].shape}")
        elif hasattr(audio_result, 'shape'):
            print(f"   📊 Shape аудио: {audio_result.shape}")
        else:
            print(f"   ⚠️  Неожиданный тип результата")
        
        # ОБРАБОТКА РЕЗУЛЬТАТА
        audio = None
        
        # 1. Если результат - список
        if isinstance(audio_result, list):
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            
            # Берем первый элемент списка
            audio = audio_result[0]
            print(f"   ✅ Использую первый элемент списка")
            
        # 2. Если результат - torch.Tensor
        elif isinstance(audio_result, torch.Tensor):
            audio = audio_result
            print(f"   ✅ Результат torch.Tensor")
            
        # 3. Если результат - tuple (старые версии)
        elif isinstance(audio_result, tuple):
            print(f"   ⚠️  Результат tuple, ищу аудио...")
            for i, item in enumerate(audio_result):
                if isinstance(item, torch.Tensor) and item.ndim in [1, 2]:
                    audio = item
                    print(f"   ✅ Найден аудио tensor в позиции {i}")
                    break
            
            if audio is None:
                # Пробуем последний элемент
                if isinstance(audio_result[-1], torch.Tensor):
                    audio = audio_result[-1]
                    print(f"   ✅ Использую последний элемент tuple")
        else:
            raise ValueError(f"Неожиданный тип результата: {type(audio_result)}")
        
        if audio is None:
            raise ValueError("Не удалось извлечь аудио из результата")
        
        # ПРОВЕРЯЕМ И ПОДГОТАВЛИВАЕМ АУДИО ДЛЯ СОХРАНЕНИЯ
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
            audio,                   # Уже правильной размерности
            target_sample_rate,
            format='wav'
        )
        
        # Проверяем что файл создан
        if not os.path.exists(temp_file.name):
            raise ValueError(f"Файл не создан: {temp_file.name}")
        
        file_size = os.path.getsize(temp_file.name)
        if file_size == 0:
            raise ValueError(f"Файл пустой: {temp_file.name}")
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / target_sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️  Время генерации: {generation_time:.2f} секунд")
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
        # Если шаблон не найден, показываем JSON
        print(f"⚠️ Шаблон index.html не найден: {e}")
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '2.0',
            'status': 'running',
            'endpoints': {
                '/': 'GET - главная страница',
                '/api/tts': 'POST - генерация аудио',
                '/api/health': 'GET - проверка здоровья',
                '/api/voices': 'GET - список голосов',
                '/api/test': 'GET - тестовый запрос',
                '/api/debug': 'GET - отладочная информация',
                '/api/status/<job_id>': 'GET - статус задачи'
            },
            'note': 'Добавьте файл templates/index.html для веб-интерфейса',
            'api_version': 'silero_v4_compatible'
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Основной endpoint для генерации TTS
    Принимает JSON: {"text": "текст", "language": "ru", "speaker": "baya"}
    """
    try:
        # Получаем и валидируем данные
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
        print(f"   🗣️  Голос: {req.speaker}")
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
            'api_note': 'Используется Silero v4 API с speakers параметром'
        }), 202
        
    except ValidationError as e:
        print(f"❌ Ошибка валидации: {e.errors()}")
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors()
        }), 400
        
    except Exception as e:
        print(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Проверка статуса задачи и получение результата"""
    try:
        print(f"📋 Проверка статуса задачи: {job_id}")
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            print(f"✅ Задача завершена: {job_id}")
            result = job.result
            
            if not result:
                print(f"❌ Задача завершена, но результат пустой")
                return jsonify({'error': 'No audio file generated'}), 500
            
            if not os.path.exists(result):
                print(f"❌ Файл не существует: {result}")
                return jsonify({'error': 'Audio file not found'}), 500
            
            file_size = os.path.getsize(result)
            if file_size == 0:
                print(f"❌ Файл пустой: {result}")
                return jsonify({'error': 'Audio file is empty'}), 500
            
            # Читаем первые несколько байт для проверки формата
            with open(result, 'rb') as f:
                header = f.read(4)
                if header == b'RIFF':
                    print(f"✅ Файл корректный WAV (RIFF заголовок)")
                else:
                    print(f"⚠️ Необычный заголовок файла: {header}")
            
            print(f"📤 Отправляю аудио файл: {result} ({file_size} bytes)")
            
            # Отправляем аудио файл
            response = send_file(
                result,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'tts_{job_id}.wav'
            )
            
            # Очистка файла после отправки
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
            print(f"❌ Задача завершилась с ошибкой: {job_id}")
            error_msg = str(job.exc_info) if job.exc_info else 'Unknown error'
            print(f"   Ошибка: {error_msg}")
            
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
            
        else:
            # Задача еще выполняется
            status = job.get_status()
            print(f"⏳ Задача выполняется: {job_id}, статус: {status}")
            
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
        print(f"❌ Ошибка при проверке статуса задачи {job_id}: {str(e)}")
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Проверка Redis
        redis_conn.ping()
        
        # Пробуем загрузить модель, если еще не загружена
        if not tts_models:
            try:
                load_tts_model('ru', 'baya')
            except Exception as e:
                print(f"⚠️ Не удалось загрузить модель при health check: {e}")
        
        # Проверяем, что модели корректно загружены
        model_status = {}
        for model_key, model_info in tts_models.items():
            model_status[model_key] = {
                'speaker_model': model_info.get('speaker_model', 'unknown'),
                'user_speaker': model_info.get('user_speaker', 'unknown'),
                'loaded_at': model_info.get('loaded_at', 'unknown'),
                'has_apply_tts': model_info.get('apply_tts') is not None
            }
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '2.0-silero-v4',
            'redis': 'connected',
            'models_loaded': list(tts_models.keys()),
            'models_count': len(tts_models),
            'model_details': model_status,
            'supported_languages': list(SILERO_CONFIG.keys()),
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available(),
            'python_version': sys.version.split()[0],
            'uptime': str(datetime.now() - startup_time),
            'cache_dir': os.environ.get('TORCH_HOME'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Ошибка health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов"""
    voices_info = {}
    
    for lang, config in SILERO_CONFIG.items():
        voices_info[lang] = []
        for speaker in config['available_speakers']:
            voice_info = {
                'id': speaker,
                'name': speaker.capitalize(),
                'description': f'{speaker} voice ({lang})',
                'language': lang,
                'sample_rate': config['sample_rate'],
                'model': config['speaker_model']
            }
            
            # Добавляем информацию о загрузке
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
    """Тестовый endpoint для проверки работы сервиса"""
    try:
        print(f"🧪 Выполняю тестовый запрос...")
        
        # Загружаем тестовую модель
        model_info = load_tts_model('ru', 'baya')
        
        if not model_info['apply_tts']:
            return jsonify({
                'success': False,
                'error': 'Модель не содержит функцию apply_tts',
                'model_info_keys': list(model_info.keys())
            }), 500
        
        # Тестовая генерация
        test_text = "Привет! Это тестовое сообщение TTS сервиса. Сервис работает на базе Silero TTS v4."
        
        print(f"   Текст: {test_text}")
        print(f"   Модель: {model_info['speaker_model']}")
        print(f"   Голос: {model_info['user_speaker']}")
        
        # Генерация аудио с проверкой обоих API
        try:
            # Пробуем современный API с speakers
            audio_result = model_info['apply_tts'](
                texts=[test_text],
                model=model_info['model'],
                sample_rate=model_info['sample_rate'],
                symbols=model_info['symbols'],
                device=model_info['device'],
                speakers=[model_info['user_speaker']]
            )
            api_type = 'modern_with_speakers'
        except TypeError:
            # Пробуем старый API
            audio_result = model_info['apply_tts'](
                texts=[test_text],
                model=model_info['model'],
                sample_rate=model_info['sample_rate'],
                symbols=model_info['symbols'],
                device=model_info['device']
            )
            api_type = 'legacy'
        
        # Обработка результата
        if isinstance(audio_result, list) and len(audio_result) > 0:
            audio = audio_result[0]
            result_type = f"list[{len(audio_result)}]"
        elif isinstance(audio_result, torch.Tensor):
            audio = audio_result
            result_type = "torch.Tensor"
        else:
            audio = None
            result_type = str(type(audio_result))
        
        audio_shape = str(audio.shape) if audio is not None and hasattr(audio, 'shape') else 'no shape'
        
        print(f"   ✅ Тест успешно завершен")
        print(f"   API тип: {api_type}")
        print(f"   Тип результата: {result_type}")
        print(f"   Формат аудио: {audio_shape}")
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'api_type': api_type,
            'result_type': result_type,
            'audio_shape': audio_shape,
            'sample_rate': model_info['sample_rate'],
            'model': model_info['speaker_model'],
            'speaker': model_info['user_speaker'],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Тестовый запрос не удался: {e}")
        print(f"Детали: {error_details}")
        
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
    # Собираем информацию о системе
    system_info = {
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version,
        'environment': {k: v for k, v in os.environ.items() 
                       if any(keyword in k for keyword in ['TORCH', 'CACHE', 'PYTHON', 'HOME'])},
    }
    
    # Проверяем кэш
    cache_info = []
    cache_dir = '/app/cache/torch/hub'
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                cache_info.append(f"{item}/ (dir)")
            else:
                cache_info.append(f"{item} ({os.path.getsize(item_path)} bytes)")
    
    return jsonify({
        'system': system_info,
        'models_loaded': list(tts_models.keys()),
        'models_detail': {k: {'speaker_model': v.get('speaker_model'), 
                             'user_speaker': v.get('user_speaker'),
                             'loaded_at': v.get('loaded_at')} 
                         for k, v in tts_models.items()} if tts_models else {},
        'cache_contents': cache_info[:20],
        'silero_config': SILERO_CONFIG,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/load-model/<language>/<speaker>', methods=['POST'])
def load_model_endpoint(language, speaker):
    """Принудительная загрузка конкретной модели"""
    try:
        model_key = f"{language}_{speaker}"
        print(f"🔄 Принудительная загрузка модели: {model_key}")
        
        if model_key in tts_models:
            print(f"   Модель уже загружена: {model_key}")
            return jsonify({
                'message': 'Model already loaded',
                'model_key': model_key,
                'loaded_at': tts_models[model_key]['loaded_at']
            })
        
        model_info = load_tts_model(language, speaker)
        
        print(f"✅ Модель успешно загружена: {model_key}")
        
        return jsonify({
            'message': 'Model loaded successfully',
            'model_key': model_key,
            'speaker_model': model_info['speaker_model'],
            'user_speaker': model_info['user_speaker'],
            'sample_rate': model_info['sample_rate'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
                        # Удаляем файлы старше 1 часа
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
    """Периодическая очистка временных файлов"""
    while True:
        time.sleep(3600)  # Каждый час
        cleanup_temp_files()

# Регистрируем очистку при завершении
atexit.register(cleanup_temp_files)

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE v2.0 - Совместимость с Silero v4/v5")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    
    # Выводим конфигурацию
    print(f"\n🔧 Конфигурация Silero:")
    for lang, config in SILERO_CONFIG.items():
        print(f"   {lang.upper()}: модель={config['speaker_model']}, "
              f"голоса={config['available_speakers']}")
    
    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Запускаем периодическую очистку в фоновом потоке
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Предварительная загрузка основной модели
    print("\n⏳ Предварительная загрузка основной модели...")
    try:
        # Сначала очищаем возможный старый кэш
        cache_path = '/app/cache/torch/hub/snakers4_silero-models_master'
        if os.path.exists(cache_path):
            print(f"🧹 Очищаю старый кэш модели...")
            shutil.rmtree(cache_path)
        
        # Загружаем модель с правильными параметрами
        model_info = load_tts_model('ru', 'baya')
        print(f"✅ Основная модель загружена")
        print(f"   Модель: {model_info['speaker_model']}")
        print(f"   Голос: {model_info['user_speaker']}")
        print(f"   Частота: {model_info['sample_rate']} Hz")
        
        # Тестируем генерацию
        print(f"\n🧪 Тестирую генерацию...")
        if model_info['apply_tts']:
            try:
                # Пробуем современный API
                test_result = model_info['apply_tts'](
                    texts=["Тестовая генерация"],
                    model=model_info['model'],
                    sample_rate=model_info['sample_rate'],
                    symbols=model_info['symbols'],
                    device=model_info['device'],
                    speakers=[model_info['user_speaker']]
                )
                print(f"✅ Тестовая генерация успешна (modern API)")
            except TypeError:
                # Пробуем старый API
                test_result = model_info['apply_tts'](
                    texts=["Тестовая генерация"],
                    model=model_info['model'],
                    sample_rate=model_info['sample_rate'],
                    symbols=model_info['symbols'],
                    device=model_info['device']
                )
                print(f"✅ Тестовая генерация успешна (legacy API)")
            
            print(f"   Тип результата: {type(test_result)}")
        else:
            print(f"⚠️ Функция apply_tts не найдена")
            
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель при старте: {e}")
        traceback.print_exc()
        print("   Модель будет загружена при первом запросе")
    
    # Запуск сервера
    print("\n🚀 Запуск Flask сервера...")
    print(f"🌐 Доступен по адресу: http://0.0.0.0:5000")
    print(f"📚 API доступен по: http://0.0.0.0:5000/api/health")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )