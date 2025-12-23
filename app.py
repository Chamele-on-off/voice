#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Упрощенная версия с прямой генерацией WAV
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import shutil
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import redis
from rq import Queue
from rq.job import Job
import threading
import atexit
import subprocess
import io

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
    db=0,
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

# ========== АЛЬТЕРНАТИВНАЯ ГЕНЕРАЦИЯ АУДИО ==========
def generate_audio_simple(text, language='ru', speaker='baya', sample_rate=16000):
    """
    Альтернативный метод генерации аудио - сохраняем файл сразу в функцию
    """
    try:
        start_time = time.time()
        print(f"\n🎵 Начинаю альтернативную генерацию аудио")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Загружаем модель
        torch.hub.set_dir('/app/cache/torch/hub')
        
        # Загружаем модель Silero
        model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=f'{speaker}_16khz' if language == 'ru' else 'lj_16khz'
        )
        
        # Генерируем аудио напрямую
        print(f"   🔊 Генерация аудио...")
        
        # Используем метод save_wav если доступен
        if hasattr(model, 'save_wav'):
            # Генерируем уникальное имя файла
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join('/app/temp_audio', filename)
            
            # Генерируем и сохраняем напрямую
            model.save_wav(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                audio_path=filepath
            )
            
            print(f"   💾 Файл сохранен: {filepath}")
            
            # Проверяем что файл создан
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   📊 Размер файла: {file_size / 1024:.1f} KB")
                
                generation_time = time.time() - start_time
                print(f"✅ Аудио успешно сгенерировано за {generation_time:.2f} секунд")
                
                # Возвращаем только имя файла
                return filename
            else:
                raise Exception("Файл не был создан")
        else:
            # Альтернативный метод через apply_tts
            print(f"   ⚠️ Метод save_wav не доступен, использую apply_tts")
            
            # Получаем полную модель
            full_model = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=f'{speaker}_16khz' if language == 'ru' else 'lj_16khz',
                verbose=False
            )
            
            # Извлекаем компоненты
            model_component = full_model[0]
            symbols = full_model[1]
            sr = full_model[2]
            example_text = full_model[3]
            apply_tts = full_model[4]
            
            # Генерируем аудио
            audio = apply_tts(
                texts=[text],
                model=model_component,
                sample_rate=sr,
                symbols=symbols,
                device=torch.device('cpu')
            )
            
            # Обрабатываем результат
            if isinstance(audio, list):
                audio_tensor = audio[0]
            else:
                audio_tensor = audio
            
            # Приводим к правильной размерности
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Создаем файл
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join('/app/temp_audio', filename)
            
            torchaudio.save(
                filepath,
                audio_tensor,
                sr,
                format='wav'
            )
            
            generation_time = time.time() - start_time
            print(f"✅ Аудио успешно сгенерировано за {generation_time:.2f} секунд")
            
            return filename
            
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Валидация входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 16000
    
    class Config:
        extra = 'forbid'

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница"""
    try:
        return render_template('index.html')
    except:
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '2.0',
            'status': 'running',
            'note': 'Use POST /api/tts with {"text": "your text"}'
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
        
        if len(req.text) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(req.text) > 5000:
            return jsonify({
                'error': f'Text too long ({len(req.text)} chars). Max is 5000.'
            }), 400
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   Текст: {req.text[:50]}...")
        
        # Создаем задачу с альтернативной функцией
        job = queue.enqueue(
            generate_audio_simple,
            args=(req.text, req.language, req.speaker, req.sample_rate),
            job_timeout=300,
            result_ttl=3600,
            failure_ttl=1800
        )
        
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача добавлена в очередь',
            'check_status': f'/api/status/{job.get_id()}',
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid request data', 'details': e.errors()}), 400
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts-direct', methods=['POST'])
def tts_direct():
    """
    Прямая генерация аудио без очереди (для тестирования)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        req = TTSRequest(**data)
        
        print(f"\n⚡ Прямая генерация аудио:")
        print(f"   Текст: {req.text[:50]}...")
        
        # Генерируем аудио напрямую
        filename = generate_audio_simple(
            req.text, 
            req.language, 
            req.speaker, 
            req.sample_rate
        )
        
        filepath = os.path.join('/app/temp_audio', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File was not created'}), 500
        
        # Отправляем файл
        return send_file(
            filepath,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'tts_{filename}'
        )
        
    except Exception as e:
        print(f"❌ Ошибка прямой генерации: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Проверка статуса задачи"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            filename = job.result
            
            if not filename:
                return jsonify({'error': 'No filename in result'}), 500
            
            # Восстанавливаем путь
            filepath = os.path.join('/app/temp_audio', filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️ Файл не найден: {filepath}")
                print(f"   Содержимое /app/temp_audio: {os.listdir('/app/temp_audio')}")
                return jsonify({'error': 'Audio file not found'}), 404
            
            print(f"📤 Отправляю файл: {filepath}")
            
            # Отправляем файл
            response = send_file(
                filepath,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'tts_{filename}'
            )
            
            # Удаляем файл после отправки
            @response.call_on_close
            def cleanup():
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"🗑️ Удален файл: {filepath}")
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
                'job_id': job_id,
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        redis_conn.ping()
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts',
            'redis': 'connected',
            'torch_version': torch.__version__,
            'python_version': sys.version.split()[0],
            'temp_audio_dir': '/app/temp_audio',
            'temp_files': len(os.listdir('/app/temp_audio')) if os.path.exists('/app/temp_audio') else 0,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/test-generate', methods=['GET'])
def test_generate():
    """Тестовая генерация"""
    try:
        test_text = "Привет, это тестовая генерация аудио."
        
        print(f"\n🧪 Тестовая генерация: {test_text}")
        
        # Генерируем тестовый файл
        filename = generate_audio_simple(test_text)
        filepath = os.path.join('/app/temp_audio', filename)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            
            # Читаем файл и конвертируем в base64 для быстрой проверки
            with open(filepath, 'rb') as f:
                audio_data = f.read()
            
            # Удаляем тестовый файл
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'message': 'Audio generated successfully',
                'filename': filename,
                'file_size_kb': round(file_size / 1024, 2),
                'file_exists': True,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'File was not created',
                'temp_dir_contents': os.listdir('/app/temp_audio') if os.path.exists('/app/temp_audio') else []
            }), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Отладочная информация"""
    temp_files = []
    if os.path.exists('/app/temp_audio'):
        temp_files = os.listdir('/app/temp_audio')
    
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version.split()[0],
        'temp_audio_dir': '/app/temp_audio',
        'temp_files_count': len(temp_files),
        'temp_files': temp_files[:10],
        'redis_connected': True if redis_conn.ping() else False,
        'queue_size': len(queue),
        'timestamp': datetime.now().isoformat()
    })

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def cleanup_temp_files():
    """Очистка временных файлов"""
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        try:
            count = 0
            for filename in os.listdir(temp_dir):
                if filename.endswith('.wav'):
                    filepath = os.path.join(temp_dir, filename)
                    if os.path.isfile(filepath):
                        try:
                            # Удаляем только старые файлы (> 1 часа)
                            file_age = time.time() - os.path.getmtime(filepath)
                            if file_age > 3600:
                                os.remove(filepath)
                                count += 1
                        except:
                            pass
            if count > 0:
                print(f"🗑️ Удалено {count} старых временных файлов")
        except Exception as e:
            print(f"⚠️ Ошибка очистки файлов: {e}")

def periodic_cleanup():
    """Периодическая очистка"""
    while True:
        time.sleep(1800)  # Каждые 30 минут
        cleanup_temp_files()

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE v2.0 - Упрощенная версия")
    print("=" * 70)
    print(f"📅 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"📁 Временные файлы: /app/temp_audio")
    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Создаем директории
    os.makedirs('/app/temp_audio', exist_ok=True)
    os.makedirs('/app/cache/torch/hub', exist_ok=True)
    
    # Запускаем очистку
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Запуск сервера
    print("\n🚀 Запуск сервера...")
    print("📚 Доступные эндпоинты:")
    print("   GET  /api/health        - Проверка здоровья")
    print("   GET  /api/test-generate - Тест генерации")
    print("   GET  /api/debug         - Отладочная информация")
    print("   POST /api/tts           - Генерация через очередь")
    print("   POST /api/tts-direct    - Прямая генерация (для тестов)")
    print("   GET  /api/status/<id>   - Статус задачи")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )