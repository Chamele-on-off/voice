#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Исправленная версия с работающим воркером
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import shutil
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import threading
import atexit
import uuid
import queue as python_queue
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
processing_queue = python_queue.Queue()
results_cache = {}
worker_running = True

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
    Возвращает кортеж из 5 элементов:
    (model, symbols, sample_rate, example_text, apply_tts)
    """
    # Формируем ключ для кэша
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
                'model': result[0],          # Модель TTS
                'symbols': result[1],        # Алфавит/символы
                'sample_rate': result[2],    # Частота дискретизации
                'example_text': result[3],   # Пример текста
                'apply_tts': result[4],      # Функция генерации
                'correct_speaker': correct_speaker,
                'device': torch.device('cpu'),
                'loaded_at': datetime.now().isoformat()
            }
            
            # Перемещаем модель на CPU
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
        logger.info(f"   Длина: {len(text)} символов")
        
        # Загружаем или получаем модель из кэша
        model_info = load_tts_model(language, speaker)
        
        # Получаем компоненты модели
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']
        
        logger.info(f"   🔊 Использую голос: {model_info['correct_speaker']}")
        logger.info(f"   🎚️  Частота: {target_sample_rate} Hz")
        
        # Генерация аудио
        audio_result = apply_tts_func(
            texts=[text],
            model=model,
            sample_rate=target_sample_rate,
            symbols=symbols,
            device=device
        )
        
        # ОБРАБОТКА РЕЗУЛЬТАТА
        # 1. Если результат - список
        if isinstance(audio_result, list):
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            
            # Берем первый элемент списка
            audio = audio_result[0]
            logger.info(f"   ✅ Использую первый элемент списка")
            
        # 2. Если результат не список
        else:
            audio = audio_result
            logger.info(f"   ✅ Результат не список, используем как есть")
        
        # ПРОВЕРЯЕМ И ПОДГОТАВЛИВАЕМ АУДИО ДЛЯ СОХРАНЕНИЯ
        logger.info(f"   🔧 Подготовка аудио для сохранения...")
        
        if not hasattr(audio, 'shape'):
            raise ValueError(f"Аудио не имеет атрибута shape. Тип: {type(audio)}")
        
        logger.info(f"   📐 Исходный shape аудио: {audio.shape}")
        
        # Приводим к правильной размерности (каналы, время)
        if audio.ndim == 1:
            # (время) -> (1, время) - один канал
            logger.info(f"   🔄 Преобразование: 1D -> 2D (добавляем канал)")
            audio = audio.unsqueeze(0) if hasattr(audio, 'unsqueeze') else audio.reshape(1, -1)
        
        logger.info(f"   📐 Финальный shape аудио: {audio.shape}")
        
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
        
        # Проверяем, что файл создан
        if not os.path.exists(filepath):
            raise ValueError(f"Файл не был создан: {filepath}")
        
        file_size = os.path.getsize(filepath)
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / target_sample_rate
        
        logger.info(f"✅ Аудио успешно сгенерировано!")
        logger.info(f"   ⏱️  Время генерации: {generation_time:.2f} секунд")
        logger.info(f"   🕒 Длительность аудио: {audio_duration:.2f} секунд")
        logger.info(f"   📁 Файл: {filename}")
        logger.info(f"   📊 Размер: {file_size / 1024:.1f} KB")
        
        return filename
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== ФУНКЦИЯ ОБРАБОТКИ ЗАДАЧ В ФОНОВОМ ПОТОКЕ ==========
def background_worker():
    """Фоновый воркер для обработки задач"""
    logger.info("🚀 Фоновый воркер запущен")
    
    while worker_running:
        try:
            # Получаем задачу из очереди (неблокирующий режим)
            try:
                task_id, text, language, speaker, sample_rate = processing_queue.get(timeout=1)
            except python_queue.Empty:
                # Очередь пуста, ждем
                time.sleep(0.1)
                continue
            
            logger.info(f"\n📋 Обрабатываю задачу {task_id}")
            logger.info(f"   Текст: '{text[:50]}...'")
            
            try:
                # Генерируем аудио
                filename = generate_audio(text, language, speaker, sample_rate)
                
                # Сохраняем результат
                results_cache[task_id] = {
                    'status': 'completed',
                    'filename': filename,
                    'completed_at': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Задача {task_id} выполнена, файл: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка выполнения задачи {task_id}: {e}")
                results_cache[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
            
            # Помечаем задачу как выполненную
            processing_queue.task_done()
            
        except Exception as e:
            logger.error(f"❌ Ошибка в фоновом воркере: {e}")
            time.sleep(1)

# ========== ЗАПУСК ФОНОВОГО ВОРКЕРА ==========
def start_background_worker():
    """Запускает фоновый воркер в отдельном потоке"""
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    logger.info("✅ Фоновый воркер запущен в отдельном потоке")
    return worker_thread

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
            'version': '2.1',
            'status': 'running',
            'background_worker': 'active',
            'endpoints': {
                '/': 'GET - главная страница',
                '/api/tts': 'POST - генерация аудио (асинхронно)',
                '/api/tts-sync': 'POST - генерация аудио (синхронно)',
                '/api/health': 'GET - проверка здоровья',
                '/api/voices': 'GET - список голосов',
                '/api/test': 'GET - тестовый запрос',
                '/api/test-generate': 'GET - тестовая генерация',
                '/api/debug': 'GET - отладочная информация',
                '/api/status/<task_id>': 'GET - статус задачи',
                '/api/queue-status': 'GET - статус очереди',
                '/api/process-task/<task_id>': 'GET - обработать задачу вручную'
            },
            'note': 'Добавьте файл templates/index.html для веб-интерфейса'
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Асинхронная генерация TTS через фоновый воркер
    ВАЖНО: Для совместимости с фронтендом возвращаем job_id вместо task_id
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
        
        logger.info(f"\n📨 Получен асинхронный TTS запрос (ID: {task_id})")
        logger.info(f"   🌐 Язык: {req.language}")
        logger.info(f"   🗣️  Голос: {req.speaker}")
        logger.info(f"   📝 Длина текста: {len(req.text)} символов")
        
        # Добавляем задачу в очередь
        processing_queue.put((task_id, req.text, req.language, req.speaker, req.sample_rate))
        
        # Инициализируем статус задачи
        results_cache[task_id] = {
            'status': 'queued',
            'queued_at': datetime.now().isoformat(),
            'queue_position': processing_queue.qsize()
        }
        
        # ВАЖНО: Для совместимости с фронтендом возвращаем job_id вместо task_id
        return jsonify({
            'job_id': task_id,  # ← Фронтенд ожидает это поле
            'task_id': task_id,  # ← Оставляем для обратной совместимости
            'status': 'queued',
            'message': 'Задача добавлена в очередь обработки',
            'estimated_time': '5-30 секунд',
            'check_status': f'/api/status/{task_id}',
            'queue_position': processing_queue.qsize(),
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
        logger.info(f"   Текст: '{req.text[:50]}...'")
        
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
        if task_id not in results_cache:
            return jsonify({'error': 'Task not found'}), 404
        
        task_info = results_cache[task_id]
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
            
            # Очистка файла после отправки и из кэша
            @response.call_on_close
            def cleanup():
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info(f"🗑️ Удален временный файл: {filepath}")
                    # Удаляем задачу из кэша
                    if task_id in results_cache:
                        del results_cache[task_id]
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
            # Задача в очереди или выполняется
            queue_position = 0
            # Подсчитываем позицию в очереди
            temp_queue = list(processing_queue.queue)
            for i, (tid, _, _, _, _) in enumerate(temp_queue):
                if tid == task_id:
                    queue_position = i + 1
                    break
            
            # Для фронтенда возвращаем job_id тоже
            return jsonify({
                'status': status,
                'job_id': task_id,  # ← Для фронтенда
                'task_id': task_id,  # ← Оригинальный ID
                'queue_position': queue_position,
                'queue_size': processing_queue.qsize(),
                'queued_at': task_info.get('queued_at'),
                'models_loaded': list(tts_models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса задачи {task_id}: {str(e)}")
        return jsonify({'error': f'Task error: {str(e)}'}), 500

@app.route('/api/process-task/<task_id>', methods=['GET'])
def process_task_manual(task_id):
    """Обработать задачу вручную (для отладки)"""
    try:
        # Ищем задачу в очереди
        task_found = False
        task_data = None
        
        temp_queue = list(processing_queue.queue)
        for tid, text, language, speaker, sample_rate in temp_queue:
            if tid == task_id:
                task_found = True
                task_data = (text, language, speaker, sample_rate)
                break
        
        if not task_found:
            # Проверяем, может задача уже в кэше
            if task_id in results_cache:
                return jsonify({
                    'message': 'Task already processed',
                    'status': results_cache[task_id]['status'],
                    'task_id': task_id
                })
            else:
                return jsonify({'error': 'Task not found in queue'}), 404
        
        logger.info(f"🛠️  Ручная обработка задачи {task_id}")
        
        # Обрабатываем задачу вручную
        text, language, speaker, sample_rate = task_data
        
        try:
            filename = generate_audio(text, language, speaker, sample_rate)
            
            # Сохраняем результат
            results_cache[task_id] = {
                'status': 'completed',
                'filename': filename,
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Задача {task_id} обработана вручную")
            
            # Создаем новую очередь без этой задачи
            new_queue = python_queue.Queue()
            for tid, t, l, s, sr in temp_queue:
                if tid != task_id:
                    new_queue.put((tid, t, l, s, sr))
            
            # Очищаем старую очередь и добавляем элементы из новой
            while not processing_queue.empty():
                try:
                    processing_queue.get_nowait()
                except python_queue.Empty:
                    break
            
            for tid, t, l, s, sr in list(new_queue.queue):
                processing_queue.put((tid, t, l, s, sr))
            
            return jsonify({
                'message': 'Task processed manually',
                'task_id': task_id,
                'filename': filename,
                'status': 'completed',
                'queue_size': processing_queue.qsize()
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка ручной обработки: {e}")
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"❌ Ошибка ручной обработки задачи: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/queue-status', methods=['GET'])
def queue_status():
    """Статус очереди"""
    queue_size = processing_queue.qsize()
    pending_tasks = list(processing_queue.queue)[:10]  # Первые 10 задач
    
    pending_ids = []
    for task_id, text, lang, speaker, _ in pending_tasks:
        pending_ids.append({
            'task_id': task_id,
            'text_preview': text[:50] + '...' if len(text) > 50 else text,
            'language': lang,
            'speaker': speaker
        })
    
    completed_tasks = {k: v for k, v in results_cache.items() if v.get('status') == 'completed'}
    failed_tasks = {k: v for k, v in results_cache.items() if v.get('status') == 'failed'}
    
    return jsonify({
        'queue_size': queue_size,
        'pending_tasks': pending_ids,
        'completed_tasks_count': len(completed_tasks),
        'failed_tasks_count': len(failed_tasks),
        'results_cache_size': len(results_cache),
        'background_worker': 'active',
        'timestamp': datetime.now().isoformat()
    })

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
        
        # Проверяем размер очереди
        queue_size = processing_queue.qsize()
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '2.1',
            'background_worker': 'active',
            'queue_size': queue_size,
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
        logger.error(f"Детали: {error_details}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_details': error_details[:500],
            'models_in_cache': list(tts_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/test-generate', methods=['GET'])
def test_generate():
    """Тестовая генерация с сохранением файла"""
    try:
        test_text = "Привет! Это тестовое сообщение TTS сервиса."
        
        logger.info(f"\n🧪 Тестовая генерация файла: {test_text}")
        
        # Используем функцию generate_audio для теста
        filename = generate_audio(test_text, 'ru', 'baya', 16000)
        filepath = os.path.join('/app/temp_audio', filename)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            
            # Читаем заголовок файла для проверки
            with open(filepath, 'rb') as f:
                header = f.read(44)  # WAV заголовок
            
            # Удаляем тестовый файл после проверки
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'message': 'Audio file generated successfully',
                'filename': filename,
                'file_size_kb': round(file_size / 1024, 2),
                'file_exists': True,
                'wav_header': header.hex()[:50] + '...',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'File was not created',
                'temp_dir': '/app/temp_audio',
                'temp_dir_exists': os.path.exists('/app/temp_audio'),
                'temp_dir_contents': os.listdir('/app/temp_audio') if os.path.exists('/app/temp_audio') else []
            }), 500
            
    except Exception as e:
        import traceback
        logger.error(f"❌ Тестовая генерация не удалась: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()[:500],
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
    
    # Получаем информацию о фоновом воркере
    worker_info = "active"
    
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version.split()[0],
        'environment': {k: v for k, v in os.environ.items() if 'TORCH' in k or 'CACHE' in k},
        'cache_dir_contents': os.listdir('/app/cache') if os.path.exists('/app/cache') else [],
        'torch_hub_cache': os.listdir('/app/cache/torch/hub') if os.path.exists('/app/cache/torch/hub') else [],
        'templates_dir': templates_dir,
        'template_files': template_files,
        'temp_audio_dir': temp_dir,
        'temp_files_count': len(temp_files),
        'temp_files': temp_files[:20],
        'models_loaded': list(tts_models.keys()),
        'tts_models_structure': {k: list(v.keys()) for k, v in tts_models.items()} if tts_models else {},
        'queue_size': processing_queue.qsize(),
        'results_cache_size': len(results_cache),
        'background_worker': worker_info,
        'worker_running': worker_running,
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
                logger.info(f"🗑️ Удалено {count} временных файлов")
        except Exception as e:
            logger.error(f"⚠️ Ошибка очистки временных файлов: {e}")

def periodic_cleanup():
    """Периодическая очистка временных файлов и кэша"""
    while True:
        time.sleep(3600)  # Каждый час
        
        # Очистка файлов
        cleanup_temp_files()
        
        # Очистка старых записей из кэша результатов
        current_time = datetime.now()
        expired_tasks = []
        
        for task_id, task_info in list(results_cache.items()):
            if task_info.get('status') in ['completed', 'failed']:
                completed_time_str = task_info.get('completed_at') or task_info.get('failed_at')
                if completed_time_str:
                    try:
                        completed_time = datetime.fromisoformat(completed_time_str)
                        if (current_time - completed_time).total_seconds() > 3600:  # 1 час
                            expired_tasks.append(task_id)
                    except:
                        pass
        
        for task_id in expired_tasks:
            del results_cache[task_id]
        
        if expired_tasks:
            logger.info(f"🗑️ Удалено {len(expired_tasks)} устаревших записей из кэша")

# Регистрируем очистку при завершении
atexit.register(cleanup_temp_files)

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE - Исправленная версия v2.1")
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
    
    # Запускаем фоновый воркер для обработки задач
    worker_thread = start_background_worker()
    
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
    print(f"📚 API доступен по: http://0.0.0.0:5000/api/health")
    print("\n📋 Доступные эндпоинты:")
    print("   POST /api/tts           - Асинхронная генерация")
    print("   POST /api/tts-sync      - Синхронная генерация (сразу файл)")
    print("   GET  /api/status/*      - Статус задачи")
    print("   GET  /api/process-task/* - Ручная обработка задачи")
    print("   GET  /api/queue-status  - Статус очереди")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )