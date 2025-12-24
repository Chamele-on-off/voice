#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Простая версия с потоками
Каждый запрос запускает синхронную генерацию в отдельном потоке
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import threading
import atexit
import uuid
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== НАСТРОЙКА ОКРУЖЕНИЯ ==========
os.environ['TORCH_HOME'] = '/app/cache'
os.environ['HF_HOME'] = '/app/cache'
os.environ['XDG_CACHE_HOME'] = '/app/cache'

# Создаем необходимые директории
os.makedirs('/app/cache/torch/hub', exist_ok=True)
os.makedirs('/app/temp_audio', exist_ok=True)

# ========== НАСТРОЙКА FLASK ==========
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
startup_time = datetime.now()
# Простое хранилище задач: task_id -> {status, filename, error}
tasks = {}

# ========== КОРРЕКТНЫЕ ИМЕНА ДИКТОРОВ SILERO ==========
SPEAKER_MAPPING = {
    'ru': {
        'baya': 'baya_16khz',
        'kseniya': 'kseniya_16khz',
        'aidar': 'aidar_16khz',
        'irina': 'irina_16khz',
        'natasha': 'natasha_16khz',
        'ruslan': 'ruslan_16khz',
    },
    'en': {
        'en_1': 'lj_16khz',
        'en_3': 'lj_16khz',
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
    """
    model_key = f"{language}_{user_speaker}"
    
    if model_key not in tts_models:
        logger.info(f"📥 Загружаю модель TTS: {language}/{user_speaker}")
        
        # Получаем правильное имя диктора
        if language in SPEAKER_MAPPING and user_speaker in SPEAKER_MAPPING[language]:
            correct_speaker = SPEAKER_MAPPING[language][user_speaker]
        else:
            # Значения по умолчанию
            if language == 'ru':
                correct_speaker = 'baya_16khz'
            else:
                correct_speaker = 'lj_16khz'
        
        logger.info(f"   Использую правильное имя: {correct_speaker}")
        
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
            
            logger.info(f"✅ Модель загружена ({len(result)} элементов)")
            
            # Сохраняем все компоненты
            tts_models[model_key] = {
                'model': result[0],
                'symbols': result[1],
                'sample_rate': result[2],
                'example_text': result[3],
                'apply_tts': result[4],
                'correct_speaker': correct_speaker,
                'device': torch.device('cpu'),
                'loaded_at': datetime.now().isoformat()
            }
            
            tts_models[model_key]['model'].to(tts_models[model_key]['device'])
            
            logger.info(f"   Sample rate: {result[2]} Hz")
            logger.info(f"   Пример текста: {result[3][:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate):
    """
    Генерация аудио из текста
    Возвращает имя сгенерированного файла
    """
    try:
        start_time = time.time()
        
        logger.info(f"\n🎵 Начинаю генерацию аудио")
        logger.info(f"   Язык: {language}, Голос: {speaker}")
        logger.info(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Загружаем или получаем модель из кэша
        model_info = load_tts_model(language, speaker)
        
        # Получаем компоненты модели
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']
        
        logger.info(f"   🔊 Использую голос: {model_info['correct_speaker']}")
        
        # Генерация аудио
        audio_result = apply_tts_func(
            texts=[text],
            model=model,
            sample_rate=target_sample_rate,
            symbols=symbols,
            device=device
        )
        
        # Обработка результата
        if isinstance(audio_result, list):
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            audio = audio_result[0]
        else:
            audio = audio_result
        
        # Приводим к правильной размерности
        if audio.ndim == 1:
            audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        
        # Создаем уникальное имя файла
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        filename = f"tts_{timestamp}_{random_id}.wav"
        filepath = os.path.join(temp_dir, filename)
        
        # Сохраняем аудио в файл
        logger.info(f"   💾 Сохраняю аудио в файл: {filepath}")
        torchaudio.save(
            filepath,
            audio,
            target_sample_rate,
            format='wav'
        )
        
        if not os.path.exists(filepath):
            raise ValueError(f"Файл не был создан: {filepath}")
        
        file_size = os.path.getsize(filepath)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Аудио успешно сгенерировано за {generation_time:.2f} секунд")
        logger.info(f"   📁 Файл: {filename}")
        
        return filename
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации аудио: {str(e)}")
        raise

# ========== ФУНКЦИЯ ВЫПОЛНЕНИЯ ЗАДАЧИ В ПОТОКЕ ==========
def execute_task(task_id, text, language, speaker, sample_rate):
    """
    Выполняет задачу TTS в отдельном потоке
    Простая синхронная генерация, но в потоке
    """
    try:
        logger.info(f"\n🧵 Запуск задачи {task_id} в потоке")
        
        # Просто вызываем синхронную генерацию
        filename = generate_audio(text, language, speaker, sample_rate)
        
        # Обновляем статус задачи
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['filename'] = filename
        tasks[task_id]['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"✅ Задача {task_id} завершена")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в задаче {task_id}: {str(e)}")
        
        # Обновляем статус задачи с ошибкой
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['failed_at'] = datetime.now().isoformat()

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница с веб-интерфейсом"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.warning(f"⚠️ Шаблон index.html не найден: {e}")
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '3.1',
            'status': 'running',
            'mode': 'threaded-sync',
            'endpoints': {
                '/': 'GET - главная страница',
                '/api/tts': 'POST - генерация в потоке',
                '/api/tts-sync': 'POST - синхронная генерация',
                '/api/health': 'GET - проверка здоровья',
                '/api/voices': 'GET - список голосов',
                '/api/test': 'GET - тестовый запрос',
                '/api/status/<task_id>': 'GET - статус задачи'
            }
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Генерация TTS в отдельном потоке
    Просто запускаем синхронную функцию в потоке
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
        
        # Генерируем уникальный ID задачи
        task_id = str(uuid.uuid4())
        
        logger.info(f"\n📨 Получен запрос для обработки в потоке (ID: {task_id})")
        logger.info(f"   Текст: '{req.text[:50]}...'")
        
        # Создаем запись о задаче
        tasks[task_id] = {
            'status': 'processing',
            'text_preview': req.text[:50] + '...' if len(req.text) > 50 else req.text,
            'language': req.language,
            'speaker': req.speaker,
            'created_at': datetime.now().isoformat(),
            'filename': None,
            'error': None
        }
        
        # Запускаем задачу в отдельном потоке
        thread = threading.Thread(
            target=execute_task,
            args=(task_id, req.text, req.language, req.speaker, req.sample_rate),
            name=f"TTS-{task_id[:8]}",
            daemon=True
        )
        
        thread.start()
        
        logger.info(f"🧵 Задача {task_id} запущена в потоке")
        
        return jsonify({
            'job_id': task_id,
            'task_id': task_id,
            'status': 'processing',
            'message': 'Задача запущена в отдельном потоке',
            'estimated_time': '5-30 секунд',
            'check_status': f'/api/status/{task_id}',
            'models_loaded': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except ValidationError as e:
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors()
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts-sync', methods=['POST'])
def tts_sync_request():
    """
    Синхронная генерация TTS (сразу возвращает файл)
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
        
        logger.info(f"\n⚡ Получен синхронный TTS запрос")
        
        # Генерируем аудио синхронно
        filename = generate_audio(req.text, req.language, req.speaker, req.sample_rate)
        filepath = os.path.join('/app/temp_audio', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File was not created'}), 500
        
        logger.info(f"📤 Отправляю файл: {filename}")
        
        # Отправляем файл
        response = send_file(
            filepath,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
        # Очистка файла после отправки
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"🗑️ Удален временный файл: {filepath}")
            except Exception as e:
                logger.error(f"⚠️ Ошибка удаления файла: {e}")
        
        return response
        
    except ValidationError as e:
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors()
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Ошибка в tts_sync_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Проверка статуса задачи"""
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task_info = tasks[task_id]
        status = task_info['status']
        
        if status == 'completed':
            # Задача выполнена
            filename = task_info['filename']
            filepath = os.path.join('/app/temp_audio', filename)
            
            if not os.path.exists(filepath):
                return jsonify({
                    'error': 'Audio file not found',
                    'status': 'completed',
                    'filename': filename
                }), 404
            
            logger.info(f"📤 Отправляю файл: {filepath}")
            
            # Отправляем файл
            response = send_file(
                filepath,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=filename
            )
            
            # Очистка файла после отправки
            @response.call_on_close
            def cleanup():
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info(f"🗑️ Удален временный файл: {filepath}")
                    # Удаляем задачу из кэша
                    if task_id in tasks:
                        del tasks[task_id]
                        logger.info(f"🗑️ Удалена задача {task_id} из кэша")
                except Exception as e:
                    logger.error(f"⚠️ Ошибка удаления файла: {e}")
            
            return response
            
        elif status == 'failed':
            # Задача завершилась с ошибкой
            return jsonify({
                'status': 'failed',
                'error': task_info.get('error', 'Unknown error'),
                'failed_at': task_info.get('failed_at'),
                'task_id': task_id
            }), 500
            
        else:
            # Задача выполняется
            return jsonify({
                'status': status,
                'job_id': task_id,
                'task_id': task_id,
                'text_preview': task_info.get('text_preview', ''),
                'language': task_info.get('language', 'ru'),
                'speaker': task_info.get('speaker', 'baya'),
                'created_at': task_info.get('created_at'),
                'models_loaded': list(tts_models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса задачи {task_id}: {str(e)}")
        return jsonify({'error': f'Task error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Пробуем загрузить модель, если еще не загружена
        if not tts_models:
            try:
                load_tts_model('ru', 'baya')
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить модель при health check: {e}")
        
        # Проверяем директорию временных файлов
        temp_files_count = len(os.listdir('/app/temp_audio')) if os.path.exists('/app/temp_audio') else 0
        
        # Считаем активные потоки
        active_threads = threading.active_count() - 1  # минус основной поток
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '3.1',
            'mode': 'threaded-sync',
            'active_threads': active_threads,
            'active_tasks': len([t for t in tasks.values() if t['status'] == 'processing']),
            'models_loaded': list(tts_models.keys()),
            'models_count': len(tts_models),
            'temp_files_count': temp_files_count,
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
            },
            {
                'id': 'aidar',
                'name': 'Айдар',
                'actual': SPEAKER_MAPPING['ru'].get('aidar', 'aidar_16khz'),
                'gender': 'male',
                'sample_rate': 16000,
                'description': 'Мужской голос'
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
        
        logger.info(f"🧪 Тестовый запрос: {test_text}")
        
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
        
        # Проверяем shape
        audio_shape = str(audio.shape) if hasattr(audio, 'shape') else 'no shape'
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'result_type': result_type,
            'audio_shape': audio_shape,
            'sample_rate': model_info['sample_rate'],
            'model_loaded': True,
            'correct_speaker': model_info['correct_speaker'],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Тестовый запрос не удался: {e}")
        
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
    # Проверяем наличие директорий
    templates_dir = '/app/templates'
    template_files = []
    if os.path.exists(templates_dir):
        template_files = os.listdir(templates_dir)
    
    # Проверяем директорию временных файлов
    temp_files = []
    temp_dir = '/app/temp_audio'
    if os.path.exists(temp_dir):
        temp_files = os.listdir(temp_dir)
    
    # Получаем информацию о потоках
    thread_info = []
    for thread in threading.enumerate():
        thread_info.append({
            'name': thread.name,
            'daemon': thread.daemon,
            'alive': thread.is_alive()
        })
    
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version.split()[0],
        'templates_dir': templates_dir,
        'template_files': template_files,
        'temp_audio_dir': temp_dir,
        'temp_files_count': len(temp_files),
        'temp_files': temp_files[:10],
        'models_loaded': list(tts_models.keys()),
        'active_tasks': len([t for t in tasks.values() if t['status'] == 'processing']),
        'completed_tasks': len([t for t in tasks.values() if t['status'] == 'completed']),
        'failed_tasks': len([t for t in tasks.values() if t['status'] == 'failed']),
        'active_threads': threading.active_count(),
        'threads': thread_info[:5],
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
                        # Удаляем только старые файлы (> 1 часа)
                        file_age = time.time() - os.path.getmtime(file_path)
                        if file_age > 3600:
                            os.remove(file_path)
                            count += 1
                    except:
                        pass
            if count > 0:
                logger.info(f"🗑️ Удалено {count} старых временных файлов")
        except Exception as e:
            logger.error(f"⚠️ Ошибка очистки временных файлов: {e}")

def periodic_cleanup():
    """Периодическая очистка временных файлов и кэша задач"""
    while True:
        time.sleep(3600)  # Каждый час
        
        # Очистка файлов
        cleanup_temp_files()
        
        # Очистка старых записей из кэша задач
        current_time = datetime.now()
        expired_tasks = []
        
        for task_id, task_info in list(tasks.items()):
            created_time_str = task_info.get('created_at')
            if created_time_str:
                try:
                    created_time = datetime.fromisoformat(created_time_str)
                    # Удаляем задачи старше 24 часов
                    if (current_time - created_time).total_seconds() > 86400:
                        expired_tasks.append(task_id)
                except:
                    pass
        
        for task_id in expired_tasks:
            del tasks[task_id]
        
        if expired_tasks:
            logger.info(f"🗑️ Удалено {len(expired_tasks)} устаревших задач из кэша")

# Регистрируем очистку при завершении
atexit.register(cleanup_temp_files)

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE - Простая версия с потоками v3.1")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    print(f"📁 Директория временных файлов: /app/temp_audio")
    
    # Проверяем наличие директории templates
    templates_dir = '/app/templates'
    if os.path.exists(templates_dir):
        print(f"✅ Директория templates существует")
        files = os.listdir(templates_dir)
        print(f"   Файлы: {files}")
    else:
        print(f"⚠️ Директория templates не существует")
        os.makedirs(templates_dir, exist_ok=True)
        print(f"   Создана новая директория")
    
    print("=" * 70)
    
    # Запускаем периодическую очистку в фоновом потоке
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    print("✅ Фоновый очиститель запущен")
    
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
        print("   Модель будет загружена при первом запросе")
    
    # Запуск сервера
    print("\n🚀 Запуск Flask сервера...")
    print(f"🌐 Доступен по адресу: http://0.0.0.0:5000")
    print(f"📚 API доступен по: http://0.0.0.0:5000/api/health")
    print("\n📋 Доступные эндпоинты:")
    print("   POST /api/tts       - Генерация в потоке (просто синхронная в отдельном потоке)")
    print("   POST /api/tts-sync  - Синхронная генерация (сразу файл)")
    print("   GET  /api/status/*  - Статус задачи")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )