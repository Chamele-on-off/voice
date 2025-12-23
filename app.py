#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Версия для старого API Silero
Очень старая версия без именованных параметров в apply_tts
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
tts_model = None
tts_symbols = None
tts_sample_rate = None
tts_example_text = None
tts_apply_tts = None
startup_time = datetime.now()
model_loaded = False

# ========== КОНФИГУРАЦИЯ SILERO TTS ==========
# Для старого API, который возвращает 5 элементов
SILERO_CONFIG = {
    'ru': {
        'speakers': ['aidar', 'baya', 'kseniya', 'irina', 'natasha', 'ruslan'],
        'sample_rate': 16000,
        'model_name': 'ru'  # Самая старая версия
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
def load_tts_model():
    """Загружает модель Silero TTS (старое API с 5 элементами)"""
    global tts_model, tts_symbols, tts_sample_rate, tts_example_text, tts_apply_tts, model_loaded
    
    if tts_model is None:
        print(f"📥 Загружаю модель Silero TTS (старое API)...")
        
        try:
            # Устанавливаем директорию кэша
            torch.hub.set_dir('/app/cache/torch/hub')
            
            # Пробуем самые старые версии
            speaker_versions = ['baya', 'aidar', 'kseniya', 'ru']
            
            for speaker_version in speaker_versions:
                try:
                    print(f"   Пробую голос/версию: {speaker_version}")
                    
                    # СТАРОЕ API: возвращает 5 элементов!
                    (tts_model, 
                     tts_symbols, 
                     tts_sample_rate, 
                     tts_example_text, 
                     tts_apply_tts) = torch.hub.load(
                        repo_or_dir='snakers4/silero-models',
                        model='silero_tts',
                        language='ru',
                        speaker=speaker_version,
                        force_reload=False,
                        trust_repo=True,
                        verbose=False
                    )
                    
                    print(f"✅ Модель успешно загружена")
                    print(f"   Sample rate: {tts_sample_rate}")
                    print(f"   Пример текста: {tts_example_text[:50]}...")
                    model_loaded = True
                    break
                    
                except ValueError as e:
                    # Ошибка распаковки (ожидали 5, получили 2)
                    print(f"   ❌ {speaker_version}: ошибка распаковки - {str(e)[:80]}")
                    # Пробуем новый API с 2 элементами
                    try:
                        tts_model, tts_example_text = torch.hub.load(
                            repo_or_dir='snakers4/silero-models',
                            model='silero_tts',
                            language='ru',
                            speaker=speaker_version,
                            force_reload=False,
                            trust_repo=True,
                            verbose=False
                        )
                        print(f"   ✅ Загружено 2 элемента (новый API)")
                        model_loaded = True
                        break
                    except Exception as e2:
                        print(f"   ❌ И новый API не сработал: {str(e2)[:80]}")
                        continue
                except Exception as e:
                    print(f"   ❌ {speaker_version}: {str(e)[:80]}")
                    continue
            
            if not model_loaded:
                raise ValueError("Не удалось загрузить модель ни в одном формате")
                
            # Тестируем вызов apply_tts
            print(f"   🔍 Тестирую вызов apply_tts...")
            
            # Вариант 1: Старый синтаксис (без именованных параметров)
            try:
                # В самом старом API apply_tts принимает позиционные аргументы:
                # apply_tts(texts, model, sample_rate, symbols, device)
                audio = tts_apply_tts(
                    texts=["Тест"],
                    model=tts_model,
                    sample_rate=tts_sample_rate if tts_sample_rate else 16000,
                    symbols=tts_symbols if tts_symbols else None,
                    device=torch.device('cpu')
                )
                print(f"   ✅ apply_tts работает с именованными параметрами")
                api_type = 'named_params'
                
            except TypeError as e:
                print(f"   ⚠️ Именованные параметры не работают: {str(e)[:80]}")
                # Вариант 2: Позиционные аргументы
                try:
                    audio = tts_apply_tts(
                        ["Тест"],           # texts
                        tts_model,          # model  
                        tts_sample_rate if tts_sample_rate else 16000,  # sample_rate
                        tts_symbols if tts_symbols else None,  # symbols
                        torch.device('cpu') # device
                    )
                    print(f"   ✅ apply_tts работает с позиционными параметрами")
                    api_type = 'positional_params'
                    
                except Exception as e2:
                    print(f"   ❌ Позиционные параметры тоже не работают: {str(e2)[:80]}")
                    # Вариант 3: Возможно apply_tts это метод модели
                    try:
                        audio = tts_model.apply_tts("Тест")
                        print(f"   ✅ model.apply_tts работает с одним аргументом")
                        api_type = 'model_method'
                    except Exception as e3:
                        print(f"   ❌ Ничего не работает")
                        raise
            
            # Сохраняем тип API для использования
            app.config['SILERO_API_TYPE'] = api_type
            print(f"   🎯 Определен тип API: {api_type}")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            traceback.print_exc()
            raise
    
    return tts_model, tts_symbols, tts_sample_rate, tts_example_text, tts_apply_tts

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """Генерация аудио из текста для старого API"""
    try:
        start_time = time.time()
        
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Загружаем модель
        model, symbols, target_sample_rate, example_text, apply_tts_func = load_tts_model()
        
        # Определяем тип API
        api_type = app.config.get('SILERO_API_TYPE', 'positional_params')
        
        print(f"   ⚙️ Вызываю apply_tts (тип API: {api_type})...")
        
        # Генерация аудио в зависимости от типа API
        if api_type == 'named_params':
            # Именованные параметры
            audio = apply_tts_func(
                texts=[text],
                model=model,
                sample_rate=target_sample_rate if target_sample_rate else 16000,
                symbols=symbols,
                device=torch.device('cpu')
            )
            
        elif api_type == 'positional_params':
            # Позиционные параметры
            audio = apply_tts_func(
                [text],  # texts
                model,   # model
                target_sample_rate if target_sample_rate else 16000,  # sample_rate
                symbols,  # symbols
                torch.device('cpu')  # device
            )
            
        elif api_type == 'model_method':
            # Метод модели
            audio = model.apply_tts(text)
        else:
            # Пробуем все варианты
            try:
                audio = apply_tts_func(
                    texts=[text],
                    model=model,
                    sample_rate=target_sample_rate if target_sample_rate else 16000,
                    symbols=symbols,
                    device=torch.device('cpu')
                )
            except TypeError:
                try:
                    audio = apply_tts_func(
                        [text],
                        model,
                        target_sample_rate if target_sample_rate else 16000,
                        symbols,
                        torch.device('cpu')
                    )
                except Exception:
                    audio = model.apply_tts(text)
        
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
        
        # Используем правильную частоту дискретизации
        save_sample_rate = target_sample_rate if target_sample_rate else 16000
        
        print(f"   💾 Сохраняю аудио в файл: {temp_file.name}")
        torchaudio.save(
            temp_file.name,
            audio,
            save_sample_rate,
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
        audio_duration = audio.shape[-1] / save_sample_rate
        
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
            'version': '6.0',
            'status': 'running',
            'model_loaded': model_loaded,
            'api_type': app.config.get('SILERO_API_TYPE', 'unknown'),
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
            'api_type': app.config.get('SILERO_API_TYPE', 'unknown'),
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
            'version': '6.0',
            'model_loaded': model_loaded,
            'api_type': app.config.get('SILERO_API_TYPE', 'unknown'),
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
        'api_type': app.config.get('SILERO_API_TYPE', 'unknown'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint"""
    try:
        print(f"🧪 Выполняю тестовый запрос...")
        
        # Загружаем модель
        model, symbols, sample_rate, example_text, apply_tts_func = load_tts_model()
        
        test_text = "Привет! Тест."
        print(f"   Текст: {test_text}")
        print(f"   API тип: {app.config.get('SILERO_API_TYPE', 'unknown')}")
        
        # Тест в зависимости от типа API
        api_type = app.config.get('SILERO_API_TYPE', 'positional_params')
        
        if api_type == 'named_params':
            audio = apply_tts_func(
                texts=[test_text],
                model=model,
                sample_rate=sample_rate if sample_rate else 16000,
                symbols=symbols,
                device=torch.device('cpu')
            )
        elif api_type == 'positional_params':
            audio = apply_tts_func(
                [test_text],
                model,
                sample_rate if sample_rate else 16000,
                symbols,
                torch.device('cpu')
            )
        elif api_type == 'model_method':
            audio = model.apply_tts(test_text)
        else:
            # Автоопределение
            try:
                audio = apply_tts_func(texts=[test_text])
            except:
                try:
                    audio = apply_tts_func([test_text])
                except:
                    audio = model.apply_tts(test_text)
        
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
            'api_type': api_type,
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
    print("🎵 ZINDAKI TTS SERVICE v6.0 - Для старого API Silero")
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
        print(f"   API тип: {app.config.get('SILERO_API_TYPE', 'unknown')}")
        
        # Тестируем
        if model_loaded:
            print(f"\n🧪 Тестирую генерацию...")
            test_text = "Тест"
            
            if app.config.get('SILERO_API_TYPE') == 'positional_params':
                model, symbols, sample_rate, example_text, apply_tts_func = load_tts_model()
                audio = apply_tts_func(
                    [test_text],
                    model,
                    sample_rate if sample_rate else 16000,
                    symbols,
                    torch.device('cpu')
                )
            else:
                # Пробуем другой вариант
                audio = apply_tts_func(texts=[test_text])
            
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