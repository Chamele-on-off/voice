#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Исправленная версия с поддержкой актуального Silero TTS API
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
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
startup_time = datetime.now()

# ========== ПОДДЕРЖИВАЕМЫЕ ГОЛОСА (БЕЗ СУФФИКСОВ) ==========
SUPPORTED_VOICES = {
    'ru': ['baya', 'aidar', 'kseniya', 'irina', 'natasha', 'ruslan'],
    'en': ['en_1']
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
    Загружает модель Silero TTS по требованию.
    Возвращает словарь с компонентами модели.
    """
    model_key = f"{language}_{user_speaker}"
    if model_key not in tts_models:
        print(f"📥 Загружаю модель TTS: {language}/{user_speaker}")
        torch.hub.set_dir('/app/cache/torch/hub')

        # Проверяем, поддерживается ли голос
        if language not in SUPPORTED_VOICES or user_speaker not in SUPPORTED_VOICES[language]:
            print(f"⚠️ Голос '{user_speaker}' не поддерживается для языка '{language}'. Используем 'baya' по умолчанию.")
            user_speaker = 'baya'

        try:
            # Загружаем модель RU v4 (стабильная версия для русских голосов)
            if language == 'ru':
                result = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker='v4_ru',  # ← ключевое изменение
                    force_reload=False,
                    trust_repo=True,
                    verbose=False
                )
            else:
                raise NotImplementedError("Only Russian is configured properly")

            print(f"✅ Модель v4_ru загружена ({len(result)} элементов)")
            model, symbols, sample_rate, example_text, apply_tts_func = result

            tts_models[model_key] = {
                'model': model,
                'symbols': symbols,
                'sample_rate': sample_rate,
                'example_text': example_text,
                'apply_tts': apply_tts_func,
                'device': torch.device('cpu'),
                'loaded_at': datetime.now().isoformat()
            }

            model.to(torch.device('cpu'))
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Пример текста: {example_text[:50]}...")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            raise

    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """
    Генерация аудио из текста.
    Возвращает путь к .wav файлу.
    """
    try:
        start_time = time.time()
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Язык: {language}, Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Длина: {len(text)} символов")

        model_info = load_tts_model(language, speaker)
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']

        print(f"   🔊 Использую голос: {speaker}")
        print(f"   🎚️  Частота: {target_sample_rate} Hz")

        # Генерация с явным указанием speakers
        audio_result = apply_tts_func(
            texts=[text],
            model=model,
            sample_rate=target_sample_rate,
            symbols=symbols,
            device=device,
            speakers=[speaker]  # ← ВАЖНО: указываем диктора здесь
        )

        # Обработка результата
        if isinstance(audio_result, list):
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            audio = audio_result[0]
        else:
            audio = audio_result

        if not hasattr(audio, 'shape'):
            raise ValueError(f"Аудио не имеет атрибута shape. Тип: {type(audio)}")

        print(f"   📐 Исходный shape аудио: {audio.shape}")

        # Приводим к (каналы, время)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        elif audio.ndim == 2:
            if audio.shape[0] != 1 and audio.shape[1] == 1:
                audio = audio.T
        else:
            raise ValueError(f"Неожиданная размерность аудио: {audio.ndim}")

        print(f"   📐 Финальный shape аудио: {audio.shape}")

        # Сохранение
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)

        torchaudio.save(temp_file.name, audio, target_sample_rate, format='wav')

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

# ========== API МАРШРУТЫ ==========
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"⚠️ Шаблон index.html не найден: {e}")
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '1.0',
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
            'note': 'Добавьте файл templates/index.html для веб-интерфейса'
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        req = TTSRequest(**data)

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
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if job.is_finished:
            result = job.result
            if not result or not os.path.exists(result):
                return jsonify({'error': 'No audio file generated'}), 500

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
            return jsonify({
                'error': 'Job failed',
                'details': error_msg,
                'status': 'failed'
            }), 500
        else:
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
    try:
        redis_conn.ping()
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
    voices_info = {
        'ru': [
            {'id': 'baya', 'name': 'Байя', 'gender': 'female', 'description': 'Чистый женский голос'},
            {'id': 'kseniya', 'name': 'Ксения', 'gender': 'female', 'description': 'Мягкий женский голос'},
            {'id': 'aidar', 'name': 'Айдар', 'gender': 'male', 'description': 'Мужской голос'},
            {'id': 'irina', 'name': 'Ирина', 'gender': 'female', 'description': 'Выразительный женский голос'},
            {'id': 'natasha', 'name': 'Наташа', 'gender': 'female', 'description': 'Молодой женский голос'},
            {'id': 'ruslan', 'name': 'Руслан', 'gender': 'male', 'description': 'Глубокий мужской голос'}
        ],
        'en': [
            {'id': 'en_1', 'name': 'English Female', 'gender': 'female', 'description': 'English female voice'}
        ]
    }

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
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    try:
        model_info = load_tts_model('ru', 'baya')
        test_text = "Привет! Это тестовое сообщение TTS сервиса."
        print(f"🧪 Тестовый запрос: {test_text}")

        audio_result = model_info['apply_tts'](
            texts=[test_text],
            model=model_info['model'],
            sample_rate=model_info['sample_rate'],
            symbols=model_info['symbols'],
            device=model_info['device'],
            speakers=['baya']
        )

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
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
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
    templates_dir = '/app/templates'
    template_files = []
    if os.path.exists(templates_dir):
        template_files = os.listdir(templates_dir)

    cache_dir = '/app/cache'
    cache_contents = os.listdir(cache_dir) if os.path.exists(cache_dir) else []

    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version,
        'environment': {k: v for k, v in os.environ.items() if 'TORCH' in k or 'CACHE' in k},
        'cache_dir_contents': cache_contents,
        'templates_dir': templates_dir,
        'template_files': template_files,
        'models_loaded': list(tts_models.keys()),
        'tts_models_structure': {k: list(v.keys()) for k, v in tts_models.items()},
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/load-model/<language>/<speaker>', methods=['POST'])
def load_model_endpoint(language, speaker):
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
            'sample_rate': model_info['sample_rate'],
            'example_text': model_info['example_text'][:100],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def cleanup_temp_files():
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
    while True:
        time.sleep(3600)
        cleanup_temp_files()

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
    print(f"📁 Директория шаблонов: /app/templates")

    templates_dir = '/app/templates'
    if os.path.exists(templates_dir):
        print(f"✅ Директория templates существует")
        files = os.listdir(templates_dir)
        print(f"   Файлы: {files}")
    else:
        print(f"⚠️ Директория templates не существует")
        os.makedirs(templates_dir, exist_ok=True)
        print(f"   Создана новая директория")

    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)

    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    print("\n⏳ Предварительная загрузка основной модели...")
    try:
        load_tts_model('ru', 'baya')
        print(f"✅ Основная модель загружена: ru_baya")
        print(f"   Частота дискретизации: {tts_models['ru_baya']['sample_rate']} Hz")
        print(f"   Пример текста: {tts_models['ru_baya']['example_text'][:50]}...")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить модель при старте: {e}")
        import traceback
        traceback.print_exc()
        print("   Модель будет загружена при первом запросе")

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