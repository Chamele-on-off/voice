#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Silero TTS с женскими голосами
Исправленная версия с правильными именами дикторов
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

# ВАЖНО: Устанавливаем переменные окружения ПЕРВЫМ ДЕЛОМ
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

# ========== ПРАВИЛЬНЫЕ ИМЕНА ДИКТОРОВ SILERO ==========
# Используем правильные имена из ошибки
CORRECT_SPEAKERS = {
    'ru': {
        'baya': 'baya_16khz',        # Исправлено: baya -> baya_16khz
        'kseniya': 'kseniya_16khz',  # Исправлено: kseniya -> kseniya_16khz
        'xenia': 'kseniya_16khz',    # xenia тоже использует kseniya_16khz
        'aidar': 'aidar_16khz',      # Мужской голос
        'irina': 'irina_16khz',      # Женский голос
        'natasha': 'natasha_16khz',  # Женский голос
        'ruslan': 'ruslan_16khz',    # Мужской голос
    },
    'en': {
        'en_1': 'lj_16khz',          # Английский женский
        'en_3': 'lj_16khz',          # Английский женский (тот же)
    }
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
    speaker: str = 'baya'  # По умолчанию используем 'baya', но преобразуем в 'baya_16khz'
    sample_rate: int = 16000  # Изменили на 16000, так как используем *_16khz модели
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
    print(f"Используем правильные имена дикторов")
    print("=" * 60)
    
    # Женские голоса для загрузки (с правильными именами)
    female_voices = [
        ('ru', 'baya_16khz'),      # Русский женский голос
        ('ru', 'kseniya_16khz'),   # Русский женский голос
        ('en', 'lj_16khz'),        # Английский женский голос
    ]
    
    loaded_count = 0
    
    for language, correct_speaker in female_voices:
        # Создаем удобный ключ для хранения
        model_key = f"{language}_{correct_speaker.replace('_16khz', '')}"
        
        try:
            print(f"\n📥 Загружаю модель: {language.upper()} - '{correct_speaker}'")
            
            # Устанавливаем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Загрузка модели
            model = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                verbose=False,
                trust_repo=True
            )
            
            # Перемещаем на CPU
            model.to('cpu')
            
            # Тестируем генерацию
            try:
                test_text = "Привет" if language == 'ru' else "Hello"
                audio = model.apply_tts(
                    text=test_text,
                    speaker=correct_speaker,
                    sample_rate=16000,
                    put_accent=True,
                    put_yo=True
                )
                test_passed = True
                print(f"   ✓ Тест генерации пройден")
            except Exception as test_error:
                print(f"   ⚠️ Тест генерации не удался: {test_error}")
                test_passed = False
            
            # Сохраняем модель
            tts_models[model_key] = {
                'model': model,
                'correct_speaker': correct_speaker,  # Храним правильное имя
                'device': 'cpu',
                'tested': test_passed,
                'sample_rate': 16000,
                'loaded_at': datetime.now().isoformat()
            }
            
            loaded_count += 1
            print(f"✅ Успешно загружено: {model_key} -> {correct_speaker}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {correct_speaker}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
    
    models_loading = False
    models_loaded = True
    
    print("\n" + "=" * 60)
    print(f"🎯 ЗАГРУЗКА МОДЕЛЕЙ ЗАВЕРШЕНА")
    print(f"   Успешно загружено: {loaded_count} из {len(female_voices)} моделей")
    print(f"   Загруженные модели: {list(tts_models.keys())}")
    print("=" * 60)

# ========== ФУНКЦИЯ ПОЛУЧЕНИЯ МОДЕЛИ ==========
def get_model(language, user_speaker):
    """Получает модель по языку и имени диктора (преобразует в правильное имя)"""
    # Преобразуем имя диктора в правильный формат
    if language in CORRECT_SPEAKERS and user_speaker in CORRECT_SPEAKERS[language]:
        correct_speaker = CORRECT_SPEAKERS[language][user_speaker]
    else:
        # По умолчанию используем baya_16khz для русского и lj_16khz для английского
        correct_speaker = 'baya_16khz' if language == 'ru' else 'lj_16khz'
    
    model_key = f"{language}_{user_speaker}"
    
    # Если модель еще не загружена
    if model_key not in tts_models:
        print(f"📥 Загружаю модель по требованию: {model_key} -> {correct_speaker}")
        
        try:
            torch.hub.set_dir('/app/cache/torch/hub')
            
            model = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                trust_repo=True
            )
            
            model.to('cpu')
            
            tts_models[model_key] = {
                'model': model,
                'correct_speaker': correct_speaker,
                'device': 'cpu',
                'sample_rate': 16000,
                'loaded_at': datetime.now().isoformat()
            }
            
            print(f"✅ Модель загружена: {model_key}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели {model_key}: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, user_speaker, sample_rate, put_accent=True, put_yo=True):
    """Генерация аудио из текста"""
    try:
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Язык: {language}, Запрошенный голос: {user_speaker}")
        
        # Получаем модель с правильным именем диктора
        model_info = get_model(language, user_speaker)
        model = model_info['model']
        correct_speaker = model_info['correct_speaker']
        target_sample_rate = model_info['sample_rate']
        
        print(f"   Использую голос: {correct_speaker}, Частота: {target_sample_rate}Hz")
        
        start_time = time.time()
        
        # Генерация аудио
        audio = model.apply_tts(
            text=text,
            speaker=correct_speaker,  # Используем ПРАВИЛЬНОЕ имя
            sample_rate=target_sample_rate,
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
        
        torchaudio.save(
            temp_file.name, 
            audio.unsqueeze(0), 
            target_sample_rate,
            format='wav'
        )
        
        generation_time = time.time() - start_time
        audio_duration = audio.shape[1] / target_sample_rate
        
        print(f"✅ Аудио успешно сгенерировано!")
        print(f"   Время: {generation_time:.2f} сек, Длительность: {audio_duration:.2f} сек")
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
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """Endpoint для озвучки"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data', 'status': 'error'}), 400
        
        req = TTSRequest(**data)
        
        # Автоматически устанавливаем sample_rate=16000 для моделей *_16khz
        if req.sample_rate != 16000:
            print(f"⚠️ Изменяю sample_rate с {req.sample_rate} на 16000 для совместимости")
            req.sample_rate = 16000
        
        # Проверяем поддержку языка
        if req.language not in ['ru', 'en']:
            return jsonify({'error': 'Unsupported language', 'status': 'error'}), 400
        
        print(f"\n📨 TTS запрос: {req.language}/{req.speaker}, текст: {len(req.text)} символов")
        
        # Создаем задачу
        job = q.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate, req.put_accent, req.put_yo),
            job_timeout=300,
            result_ttl=3600
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача добавлена в очередь',
            'models_available': list(tts_models.keys()),
            'speaker_mapping': CORRECT_SPEAKERS[req.language] if req.language in CORRECT_SPEAKERS else {}
        }), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid data', 'details': e.errors()}), 400
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return jsonify({'error': 'Internal error', 'message': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result and os.path.exists(result):
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
                    except:
                        pass
                
                return response
            else:
                return jsonify({'error': 'No audio file'}), 500
                
        elif job.is_failed:
            return jsonify({'error': 'Job failed', 'details': str(job.exc_info)}), 500
            
        else:
            return jsonify({
                'status': job.get_status(),
                'models_loaded': list(tts_models.keys())
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья"""
    try:
        redis_conn.ping()
        
        return jsonify({
            'status': 'healthy',
            'redis': 'connected',
            'models_loaded': list(tts_models.keys()),
            'models_loaded_count': len(tts_models),
            'models_loading': models_loading,
            'torch_version': torch.__version__,
            'torch_available': torch.cuda.is_available(),
            'cache_dir': os.environ.get('TORCH_HOME'),
            'service': 'zindaki-tts-female-corrected',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'models_loaded': list(tts_models.keys())
        }), 500

@app.route('/api/load-models', methods=['POST'])
def force_load_models():
    """Принудительная загрузка моделей"""
    if models_loading:
        return jsonify({'message': 'Already loading'}), 200
    
    thread = threading.Thread(target=load_all_models)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Loading started',
        'existing_models': list(tts_models.keys())
    })

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Список доступных голосов"""
    voices_info = {
        'ru': [
            {'id': 'baya', 'name': 'Байя', 'actual': 'baya_16khz', 'sample_rate': 16000},
            {'id': 'kseniya', 'name': 'Ксения', 'actual': 'kseniya_16khz', 'sample_rate': 16000},
            {'id': 'aidar', 'name': 'Айдар', 'actual': 'aidar_16khz', 'sample_rate': 16000},
            {'id': 'irina', 'name': 'Ирина', 'actual': 'irina_16khz', 'sample_rate': 16000},
        ],
        'en': [
            {'id': 'en_1', 'name': 'English Female', 'actual': 'lj_16khz', 'sample_rate': 16000},
        ]
    }
    
    return jsonify({
        'voices': voices_info,
        'loaded': list(tts_models.keys()),
        'speaker_mapping': CORRECT_SPEAKERS
    })

# ========== ЗАПУСК ==========
if __name__ == '__main__':
    # Создаем директории
    os.makedirs('/app/temp_audio', exist_ok=True)
    os.makedirs('/app/cache', exist_ok=True)
    os.makedirs('/app/cache/torch/hub', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🎵 ZINDAKI TTS SERVICE - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 60)
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"📁 Кэш: {os.environ.get('TORCH_HOME')}")
    print("=" * 60)
    
    # Загружаем модели
    print("\n⏳ Загружаю модели...")
    load_all_models()
    
    # Запускаем сервер
    print("\n🚀 Запускаю сервер...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )