#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Полная исправленная версия
Исправлена ошибка: 'list' object has no attribute 'unsqueeze'
Функция apply_tts возвращает список аудиофайлов!
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import redis
from rq import Queue
from rq.job import Job
import threading
import atexit

# ========== НАСТРОЙКА ОКРУЖЕНИЯ ==========
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем необходимые директории
os.makedirs('/app/cache/torch/hub', exist_ok=True)
os.makedirs('/app/temp_audio', exist_ok=True)

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
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
startup_time = datetime.now()

# ========== КОРРЕКТНЫЕ ИМЕНА ДИКТОРОВ SILERO ==========
# Используем правильные имена из доступных списков
SPEAKER_MAPPING = {
    'ru': {
        'baya': 'baya_16khz',      # Русский женский голос
        'kseniya': 'kseniya_16khz', # Русский женский голос
        'aidar': 'aidar_16khz',     # Русский мужской голос
        'irina': 'irina_16khz',     # Русский женский голос
        'natasha': 'natasha_16khz', # Русский женский голос
        'ruslan': 'ruslan_16khz',   # Русский мужской голос
    },
    'en': {
        'en_1': 'lj_16khz',         # Английский женский голос
        'en_3': 'lj_16khz',         # Английский женский голос (тот же)
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
    Загружает модель Silero TTS по требованию
    Возвращает кортеж из 5 элементов:
    (model, symbols, sample_rate, example_text, apply_tts)
    """
    # Формируем ключ для кэша
    model_key = f"{language}_{user_speaker}"
    
    if model_key not in tts_models:
        print(f"📥 Загружаю модель TTS: {language}/{user_speaker}")
        
        # Получаем правильное имя диктора
        if language in SPEAKER_MAPPING and user_speaker in SPEAKER_MAPPING[language]:
            correct_speaker = SPEAKER_MAPPING[language][user_speaker]
        else:
            # Значения по умолчанию
            if language == 'ru':
                correct_speaker = 'baya_16khz'
            else:
                correct_speaker = 'lj_16khz'
        
        print(f"   Использую правильное имя: {correct_speaker}")
        
        # Устанавливаем директорию кэша
        torch.hub.set_dir('/app/cache/torch/hub')
        
        try:
            # Загружаем модель с force_reload=False для использования кэша
            result = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            
            print(f"✅ Модель загружена ({len(result)} элементов)")
            
            # Сохраняем все компоненты
            tts_models[model_key] = {
                'model': result[0],          # Модель TTS
                'symbols': result[1],        # Алфавит/символы
                'sample_rate': result[2],    # Частота дискретизации
                'example_text': result[3],   # Пример текста
                'apply_tts': result[4],      # Функция генерации
                'correct_speaker': correct_speaker,  # Правильное имя диктора
                'device': torch.device('cpu'),       # Устройство
                'loaded_at': datetime.now().isoformat()
            }
            
            # Перемещаем модель на CPU
            tts_models[model_key]['model'].to(tts_models[model_key]['device'])
            
            print(f"   Sample rate: {result[2]} Hz")
            print(f"   Пример текста: {result[3][:50]}...")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """
    Генерация аудио из текста
    ВАЖНО: функция apply_tts возвращает СПИСОК аудиофайлов!
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
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']
        
        print(f"   🔊 Использую голос: {model_info['correct_speaker']}")
        print(f"   🎚️  Частота: {target_sample_rate} Hz")
        
        # Генерация аудио
        # ВАЖНО: apply_tts возвращает СПИСОК аудиофайлов для каждого текста!
        audio_result = apply_tts_func(
            texts=[text],           # Список текстов!
            model=model,            # Модель TTS
            sample_rate=target_sample_rate,  # Частота дискретизации
            symbols=symbols,        # Алфавит/символы
            device=device           # Устройство (CPU)
        )
        
        # ДИАГНОСТИКА: выводим тип результата
        print(f"   📊 Тип результата apply_tts: {type(audio_result)}")
        if isinstance(audio_result, list):
            print(f"   📊 Длина списка: {len(audio_result)}")
            if len(audio_result) > 0:
                print(f"   📊 Тип первого элемента: {type(audio_result[0])}")
                if hasattr(audio_result[0], 'shape'):
                    print(f"   📊 Shape первого элемента: {audio_result[0].shape}")
        
        # ОБРАБОТКА РЕЗУЛЬТАТА
        # 1. Если результат - список
        if isinstance(audio_result, list):
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            
            # Берем первый элемент списка
            audio = audio_result[0]
            print(f"   ✅ Использую первый элемент списка")
            
        # 2. Если результат не список
        else:
            audio = audio_result
            print(f"   ✅ Результат не список, используем как есть")
        
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
            if audio.shape[0] != 1:
                # Возможно (время, каналы) -> (каналы, время)
                print(f"   🔄 Проверяем ориентацию каналов...")
                # Оставляем как есть, torchaudio разберется
                pass
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
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / target_sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️  Время генерации: {generation_time:.2f} секунд")
        print(f"   🕒 Длительность аудио: {audio_duration:.2f} секунд")
        print(f"   📁 Файл: {temp_file.name}")
        print(f"   📊 Размер: {os.path.getsize(temp_file.name) / 1024:.1f} KB")
        
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== УПРОЩЕННАЯ ВЕРСИЯ ГЕНЕРАЦИИ (альтернативная) ==========
def generate_audio_simple(text, language, speaker, sample_rate):
    """
    Упрощенная версия генерации аудио
    Для случаев, когда основная функция не работает
    """
    try:
        print(f"🎵 Упрощенная генерация: {language}/{speaker}")
        
        # Загружаем модель
        model_info = load_tts_model(language, speaker)
        
        # Генерация
        audio_result = model_info['apply_tts'](
            texts=[text],
            model=model_info['model'],
            sample_rate=model_info['sample_rate'],
            symbols=model_info['symbols'],
            device=model_info['device']
        )
        
        # Обработка результата
        if isinstance(audio_result, list):
            print(f"   📊 Результат - список из {len(audio_result)} элементов")
            audio = audio_result[0]
        else:
            print(f"   📊 Результат - не список: {type(audio_result)}")
            audio = audio_result
        
        # Приводим к правильной размерности
        if hasattr(audio, 'ndim'):
            if audio.ndim == 1:
                # 1D -> 2D
                audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        
        # Сохраняем
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        torchaudio.save(
            temp_file.name,
            audio,
            model_info['sample_rate'],
            format='wav'
        )
        
        print(f"✅ Аудио сохранено: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Ошибка в упрощенной генерации: {e}")
        # Пробуем основную функцию
        return generate_audio(text, language, speaker, sample_rate)

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'service': 'Zindaki TTS Service',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/api/tts': 'POST - генерация аудио',
            '/api/health': 'GET - проверка здоровья',
            '/api/voices': 'GET - список голосов',
            '/api/test': 'GET - тестовый запрос',
            '/api/debug': 'GET - отладочная информация'
        }
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
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   🌐 Язык: {req.language}")
        print(f"   🗣️  Голос: {req.speaker}")
        print(f"   📝 Длина текста: {len(req.text)} символов")
        
        # Создаем фоновую задачу
        job = queue.enqueue(
            generate_audio,  # Используем основную функцию
            args=(req.text, req.language, req.speaker, req.sample_rate),
            job_timeout=300,    # 5 минут таймаут
            result_ttl=3600,    # Результат хранится 1 час
            failure_ttl=1800    # Информация о неудачных задачах 30 минут
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача добавлена в очередь обработки',
            'estimated_time': '5-30 секунд',
            'models_loaded': list(tts_models.keys()),
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
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            
            if not result or not os.path.exists(result):
                return jsonify({'error': 'No audio file generated'}), 500
            
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
            error_msg = str(job.exc_info) if job.exc_info else 'Unknown error'
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
            
        else:
            # Задача еще выполняется
            return jsonify({
                'status': job.get_status(),
                'position': job.get_position() if hasattr(job, 'get_position') else 'unknown',
                'models_loaded': list(tts_models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
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
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'redis': 'connected',
            'models_loaded': list(tts_models.keys()),
            'models_count': len(tts_models),
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available(),
            'python_version': sys.version.split()[0],
            'uptime': str(datetime.now() - startup_time),
            'cache_dir': os.environ.get('TORCH_HOME'),
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
    """Возвращает список доступных голосов"""
    voices_info = {
        'ru': [
            {
                'id': 'baya',
                'name': 'Байя',
                'actual': SPEAKER_MAPPING['ru'].get('baya', 'baya_16khz'),
                'gender': 'female',
                'sample_rate': 16000,
                'description': 'Чистый женский голос'
            },
            {
                'id': 'kseniya',
                'name': 'Ксения',
                'actual': SPEAKER_MAPPING['ru'].get('kseniya', 'kseniya_16khz'),
                'gender': 'female',
                'sample_rate': 16000,
                'description': 'Мягкий женский голос'
            }
        ],
        'en': [
            {
                'id': 'en_1',
                'name': 'English Female',
                'actual': SPEAKER_MAPPING['en'].get('en_1', 'lj_16khz'),
                'gender': 'female',
                'sample_rate': 16000,
                'description': 'English female voice'
            }
        ]
    }
    
    # Фильтруем только загруженные голоса
    loaded_voices = {}
    for lang in voices_info:
        loaded_voices[lang] = [
            voice for voice in voices_info[lang]
            if f"{lang}_{voice['id']}" in tts_models
        ]
    
    return jsonify({
        'all_voices': voices_info,
        'loaded_voices': loaded_voices,
        'total_loaded': len(tts_models),
        'speaker_mapping': SPEAKER_MAPPING,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint для проверки работы сервиса"""
    try:
        # Загружаем тестовую модель
        model_info = load_tts_model('ru', 'baya')
        
        # Тестовая генерация
        test_text = "Привет! Это тестовое сообщение TTS сервиса."
        
        print(f"🧪 Тестовый запрос: {test_text}")
        
        # Генерация аудио
        audio_result = model_info['apply_tts'](
            texts=[test_text],
            model=model_info['model'],
            sample_rate=model_info['sample_rate'],
            symbols=model_info['symbols'],
            device=model_info['device']
        )
        
        # Обработка результата
        if isinstance(audio_result, list):
            audio = audio_result[0]
            result_type = f"list[{len(audio_result)}]"
        else:
            audio = audio_result
            result_type = str(type(audio_result))
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'result_type': result_type,
            'audio_shape': str(audio.shape) if hasattr(audio, 'shape') else 'no shape',
            'sample_rate': model_info['sample_rate'],
            'model_loaded': True,
            'correct_speaker': model_info['correct_speaker'],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
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
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version,
        'environment': {k: v for k, v in os.environ.items() if 'TORCH' in k or 'CACHE' in k},
        'cache_dir_contents': os.listdir('/app/cache') if os.path.exists('/app/cache') else [],
        'models_loaded': list(tts_models.keys()),
        'tts_models_structure': {k: list(v.keys()) for k, v in tts_models.items()} if tts_models else {},
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/load-model/<language>/<speaker>', methods=['POST'])
def load_model_endpoint(language, speaker):
    """Принудительная загрузка конкретной модели"""
    try:
        model_key = f"{language}_{speaker}"
        
        if model_key in tts_models:
            return jsonify({
                'message': 'Model already loaded',
                'model_key': model_key,
                'loaded_at': tts_models[model_key]['loaded_at']
            })
        
        model_info = load_tts_model(language, speaker)
        
        return jsonify({
            'message': 'Model loaded successfully',
            'model_key': model_key,
            'correct_speaker': model_info['correct_speaker'],
            'sample_rate': model_info['sample_rate'],
            'example_text': model_info['example_text'][:100],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def cleanup_temp_files():
    """Очистка временных файлов"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            count = 0
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        count += 1
                    except:
                        pass
            if count > 0:
                print(f"🗑️ Удалено {count} временных файлов")
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
    print("🎵 ZINDAKI TTS SERVICE - Сервис озвучки для онлайн-школы")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Запускаем периодическую очистку в фоновом потоке
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Предварительная загрузка основной модели
    print("\n⏳ Предварительная загрузка основной модели...")
    try:
        load_tts_model('ru', 'baya')
        print(f"✅ Основная модель загружена: ru_baya")
        print(f"   Используется голос: {tts_models['ru_baya']['correct_speaker']}")
        print(f"   Частота дискретизации: {tts_models['ru_baya']['sample_rate']} Hz")
        print(f"   Пример текста: {tts_models['ru_baya']['example_text'][:50]}...")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель при старте: {e}")
        import traceback
        traceback.print_exc()
        print("   Модель будет загружена при первом запросе")
    
    # Запуск сервера
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