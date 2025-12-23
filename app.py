import os
import uuid
import torch
import io
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from rq import Queue
from rq.job import Job
import redis
import torchaudio
import tempfile
import atexit
import shutil
import threading
import time

# ========== НАСТРОЙКА ПЕРЕД ВСЕМ ==========
# Устанавливаем переменные окружения для кэша ДО импортов torch
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем директории кэша
os.makedirs('/app/cache', exist_ok=True)
os.makedirs('/app/cache/torch/hub', exist_ok=True)

# ========== ИНИЦИАЛИЗАЦИЯ REDIS ==========
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'tts-redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1,
    socket_connect_timeout=5,
    socket_timeout=5
)
q = Queue(connection=redis_conn, default_timeout=600)

# ========== МОДЕЛИ SILERO ==========
MODELS = {
    'ru': 'v3_ru',
    'en': 'v3_en'
}

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
models_loading = False
models_loaded = False

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Модель для валидации входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'  # Женский голос по умолчанию
    sample_rate: int = 24000
    put_accent: bool = True
    put_yo: bool = True

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛЕЙ ==========
def load_all_models():
    """Загрузка всех женских моделей при старте"""
    global tts_models, models_loading, models_loaded
    
    models_loading = True
    print("=" * 50)
    print("🚀 НАЧАЛО ЗАГРУЗКИ МОДЕЛЕЙ SILERO TTS")
    print("=" * 50)
    
    # Женские голоса для загрузки
    female_voices = [
        ('ru', 'baya'),      # Русский женский 1
        ('ru', 'kseniya'),   # Русский женский 2  
        ('ru', 'xenia'),     # Русский женский 3
        ('en', 'en_1'),      # Английский женский 1
        ('en', 'en_3'),      # Английский женский 2
    ]
    
    loaded_count = 0
    
    for language, speaker in female_voices:
        model_key = f"{language}_{speaker}"
        
        try:
            print(f"\n📥 Загружаю модель: {language} - {speaker}")
            
            # Загрузка через torch.hub
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=speaker,
                force_reload=False,
                verbose=True
            )
            
            # Перемещаем на CPU
            model.to('cpu')
            
            # Сохраняем в кэш
            tts_models[model_key] = {
                'model': model,
                'example_text': example_text,
                'device': 'cpu'
            }
            
            loaded_count += 1
            print(f"✅ Успешно загружено: {model_key}")
            
            # Тестируем генерацию
            test_text = "Тест" if language == 'ru' else "Test"
            try:
                audio = model.apply_tts(
                    text=test_text,
                    speaker=speaker,
                    sample_rate=24000
                )
                print(f"   ✓ Тест генерации пройден")
            except:
                print(f"   ⚠️ Тест генерации не удался")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_key}: {str(e)}")
            print(f"   Пробую альтернативный метод...")
            
            try:
                # Альтернативная попытка
                model = torch.hub.load(
                    'snakers4/silero-models',
                    'silero_tts',
                    language=language,
                    speaker=speaker
                )
                model.to('cpu')
                tts_models[model_key] = {'model': model, 'device': 'cpu'}
                loaded_count += 1
                print(f"✅ Загружено (альтернативный метод): {model_key}")
            except Exception as e2:
                print(f"❌ Полный провал: {e2}")
    
    models_loading = False
    models_loaded = True
    
    print("\n" + "=" * 50)
    print(f"🎯 ЗАГРУЗКА ЗАВЕРШЕНА")
    print(f"   Успешно загружено: {loaded_count} из {len(female_voices)} моделей")
    print(f"   Загруженные модели: {list(tts_models.keys())}")
    print("=" * 50)

# ========== ФУНКЦИЯ ЗАГРУЗКИ КОНКРЕТНОЙ МОДЕЛИ ==========
def load_model(language, speaker):
    """Загружаем конкретную модель Silero TTS"""
    model_key = f"{language}_{speaker}"
    
    if model_key not in tts_models:
        print(f"📥 Загружаю модель по требованию: {model_key}")
        
        try:
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=speaker,
                force_reload=False
            )
            model.to('cpu')
            
            tts_models[model_key] = {
                'model': model,
                'example_text': example_text,
                'device': 'cpu'
            }
            print(f"✅ Модель загружена: {model_key}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {model_key}: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate, put_accent=True, put_yo=True):
    """Генерация аудио (вызывается из фоновой задачи)"""
    try:
        print(f"\n🎵 Генерация аудио: '{text[:100]}...'")
        print(f"   Язык: {language}, Голос: {speaker}")
        
        start_time = time.time()
        
        # Загружаем модель
        model_info = load_model(language, speaker)
        model = model_info['model']
        
        # Генерация аудио
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo
        )
        
        # Сохраняем во временный файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            delete=False,
            dir=temp_dir
        )
        
        # Сохраняем аудио
        torchaudio.save(
            temp_file.name, 
            audio.unsqueeze(0), 
            sample_rate,
            format='wav'
        )
        
        generation_time = time.time() - start_time
        print(f"✅ Аудио сгенерировано за {generation_time:.2f} секунд")
        print(f"   Файл: {temp_file.name}")
        
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
    """Основной endpoint для запроса озвучки"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        req = TTSRequest(**data)
        
        # Проверяем язык
        if req.language not in MODELS:
            return jsonify({'error': f'Unsupported language: {req.language}'}), 400
        
        # Создаем фоновую задачу
        job = q.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate, req.put_accent, req.put_yo),
            job_timeout=300,
            result_ttl=3600
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Task queued for processing',
            'estimated_time': '10-30 seconds',
            'models_available': list(tts_models.keys())
        }), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid data', 'details': e.errors()}), 400
    except Exception as e:
        print(f"Error in tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({'error': 'Job result is empty'}), 500
            
            # Проверяем, является ли результат путем к файлу
            if isinstance(result, str) and os.path.exists(result):
                try:
                    # Отправляем файл
                    response = send_file(
                        result,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'tts_{job_id}.wav'
                    )
                    
                    # Удаляем файл после отправки (в фоне)
                    @response.call_on_close
                    def cleanup_file():
                        try:
                            if os.path.exists(result):
                                os.remove(result)
                        except:
                            pass
                    
                    return response
                except Exception as e:
                    print(f"Error sending file: {str(e)}")
                    return jsonify({'error': 'Error sending audio file'}), 500
            else:
                return jsonify({'error': 'Invalid job result'}), 500
                
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
                'models_loaded': list(tts_models.keys())
            }), 200
            
    except Exception as e:
        print(f"Error in get_status: {str(e)}")
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов"""
    voices = {
        'ru': ['baya', 'kseniya', 'xenia'],  # Женские русские
        'en': ['en_1', 'en_3']  # Женские английские
    }
    
    # Фильтруем только загруженные голоса
    available_voices = {}
    for lang, speakers in voices.items():
        available_voices[lang] = [
            s for s in speakers 
            if f"{lang}_{s}" in tts_models
        ]
    
    return jsonify({
        'all_voices': voices,
        'loaded_voices': available_voices,
        'total_loaded': len(tts_models)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Проверяем Redis соединение
        redis_conn.ping()
        
        return jsonify({
            'status': 'healthy',
            'redis': 'connected',
            'models_loaded': list(tts_models.keys()),
            'models_loading': models_loading,
            'models_loaded_count': len(tts_models),
            'queue_size': len(q),
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            'cache_dir': os.environ.get('TORCH_HOME'),
            'service': 'zindaki-tts-female'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys()),
            'torch_version': torch.__version__ if 'torch' in globals() else 'not loaded'
        }), 500

@app.route('/api/load-models', methods=['POST'])
def force_load_models_endpoint():
    """Принудительная загрузка моделей"""
    if models_loading:
        return jsonify({'message': 'Models are already loading'}), 200
    
    thread = threading.Thread(target=load_all_models)
    thread.start()
    
    return jsonify({
        'message': 'Model loading started',
        'loading': True,
        'existing_models': list(tts_models.keys())
    })

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def cleanup_temp_files():
    """Очистка временных файлов при завершении"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        except Exception as e:
            print(f"Error cleaning temp dir: {e}")

# Регистрируем очистку при завершении
atexit.register(cleanup_temp_files)

# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========
if __name__ == '__main__':
    # Создаем необходимые директории
    os.makedirs('/app/temp_audio', exist_ok=True)
    os.makedirs('/app/cache', exist_ok=True)
    os.makedirs('/app/cache/torch/hub', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🎵 ZINDAKI TTS SERVICE - FEMALE VOICES EDITION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchAudio version: {torchaudio.__version__}")
    print(f"Cache directory: {os.environ.get('TORCH_HOME')}")
    print("=" * 60)
    
    # Запускаем загрузку моделей в фоне
    print("\n⏳ Starting model loading in background thread...")
    load_thread = threading.Thread(target=load_all_models, daemon=True)
    load_thread.start()
    
    # Запускаем Flask приложение
    print("🚀 Starting Flask server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )