#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Версия с кэшированием фраз
Каждый запрос проверяется в кэше перед генерацией
"""

import os
import sys
import torch
import torchaudio
import tempfile
import time
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import threading
import atexit
import uuid
import logging
import hashlib
import sqlite3
import shutil
from pathlib import Path

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
os.makedirs('/app/tts_cache/audio', exist_ok=True)
os.makedirs('/app/tts_cache/db', exist_ok=True)

# ========== НАСТРОЙКА FLASK ==========
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ==========
tts_models = {}
startup_time = datetime.now()
max_concurrent_threads = 50
cache_hits = 0
cache_misses = 0

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

# ========== КЭШ TTS ==========
class TTSCache:
    """Кэш для TTS фраз с использованием SQLite"""
    
    def __init__(self):
        self.cache_dir = '/app/tts_cache'
        self.audio_dir = os.path.join(self.cache_dir, 'audio')
        self.db_path = os.path.join(self.cache_dir, 'db', 'tts_cache.db')
        
        # Создаем директории если не существуют
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._init_database()
        self.max_cache_size_mb = 1024  # 1 GB максимальный размер кэша
        self.cache_ttl_days = 30  # Храним 30 дней
        
        logger.info(f"✅ Кэш инициализирован: {self.db_path}")
    
    def _init_database(self):
        """Инициализация базы данных SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создаем таблицу если не существует
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tts_cache (
                cache_key TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                language TEXT NOT NULL,
                speaker TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                duration_sec REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                generation_time REAL
            )
        ''')
        
        # Создаем индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_hash ON tts_cache(text_hash, language, speaker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON tts_cache(last_accessed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON tts_cache(created_at)')
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(self, text, language, speaker, sample_rate):
        """Генерация уникального ключа для кэша"""
        # Нормализуем текст: убираем лишние пробелы, приводим к нижнему регистру
        normalized_text = ' '.join(text.strip().split()).lower()
        
        # Создаем хеш из текста и параметров
        text_hash = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
        params_hash = hashlib.md5(f"{language}_{speaker}_{sample_rate}".encode('utf-8')).hexdigest()
        
        return f"{text_hash}_{params_hash}"
    
    def _generate_text_hash(self, text):
        """Генерация хеша только для текста"""
        normalized_text = ' '.join(text.strip().split()).lower()
        return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
    
    def get(self, text, language, speaker, sample_rate):
        """Получение аудио из кэша"""
        cache_key = self._generate_cache_key(text, language, speaker, sample_rate)
        text_hash = self._generate_text_hash(text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT file_path, duration_sec, generation_time 
                FROM tts_cache 
                WHERE cache_key = ? OR (text_hash = ? AND language = ? AND speaker = ? AND sample_rate = ?)
                LIMIT 1
            ''', (cache_key, text_hash, language, speaker, sample_rate))
            
            result = cursor.fetchone()
            
            if result:
                file_path, duration_sec, generation_time = result
                
                # Проверяем существование файла
                if os.path.exists(file_path):
                    # Обновляем статистику доступа
                    cursor.execute('''
                        UPDATE tts_cache 
                        SET last_accessed = CURRENT_TIMESTAMP, 
                            access_count = access_count + 1
                        WHERE cache_key = ?
                    ''', (cache_key,))
                    conn.commit()
                    
                    logger.info(f"✅ Кэш хит: {text[:50]}...")
                    return {
                        'hit': True,
                        'file_path': file_path,
                        'duration_sec': duration_sec,
                        'generation_time': generation_time,
                        'cached': True
                    }
        
        except Exception as e:
            logger.error(f"❌ Ошибка при получении из кэша: {e}")
        finally:
            conn.close()
        
        return {'hit': False}
    
    def put(self, text, language, speaker, sample_rate, audio_filepath, generation_time):
        """Добавление аудио в кэш"""
        cache_key = self._generate_cache_key(text, language, speaker, sample_rate)
        text_hash = self._generate_text_hash(text)
        
        # Создаем уникальное имя файла в кэше
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_filename = f"{cache_key}_{timestamp}.wav"
        cache_filepath = os.path.join(self.audio_dir, cache_filename)
        
        try:
            # Копируем файл в кэш
            shutil.copy2(audio_filepath, cache_filepath)
            file_size = os.path.getsize(cache_filepath)
            
            # Пытаемся определить длительность аудио
            duration_sec = 0
            try:
                info = torchaudio.info(cache_filepath)
                duration_sec = info.num_frames / info.sample_rate
            except:
                pass
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Вставляем или заменяем запись
            cursor.execute('''
                INSERT OR REPLACE INTO tts_cache 
                (cache_key, text_hash, text, language, speaker, sample_rate, 
                 file_path, file_size, duration_sec, generation_time, 
                 created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            ''', (cache_key, text_hash, text[:1000], language, speaker, sample_rate,
                  cache_filepath, file_size, duration_sec, generation_time))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Добавлено в кэш: {text[:50]}... (ключ: {cache_key[:16]}...)")
            
            # Очищаем старый кэш если нужно
            self._cleanup_old_cache()
            
            return cache_filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении в кэш: {e}")
            return None
    
    def _cleanup_old_cache(self):
        """Очистка старого кэша"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Удаляем записи старше TTL
            cutoff_date = datetime.now() - timedelta(days=self.cache_ttl_days)
            cursor.execute('''
                SELECT cache_key, file_path FROM tts_cache 
                WHERE created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            old_records = cursor.fetchall()
            for cache_key, file_path in old_records:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    cursor.execute('DELETE FROM tts_cache WHERE cache_key = ?', (cache_key,))
                    logger.debug(f"🗑️ Удален старый кэш: {cache_key[:16]}...")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка удаления старого кэша: {e}")
            
            conn.commit()
            
            # 2. Проверяем общий размер кэша
            cursor.execute('SELECT SUM(file_size) FROM tts_cache')
            total_size_bytes = cursor.fetchone()[0] or 0
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            if total_size_mb > self.max_cache_size_mb:
                logger.info(f"📊 Размер кэша: {total_size_mb:.1f} MB (макс: {self.max_cache_size_mb} MB)")
                
                # Удаляем наименее используемые записи
                cursor.execute('''
                    SELECT cache_key, file_path, access_count, last_accessed 
                    FROM tts_cache 
                    ORDER BY access_count ASC, last_accessed ASC
                ''')
                
                for cache_key, file_path, access_count, last_accessed in cursor.fetchall():
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        cursor.execute('DELETE FROM tts_cache WHERE cache_key = ?', (cache_key,))
                        logger.debug(f"🗑️ Удален редко используемый кэш: {cache_key[:16]}... (использований: {access_count})")
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка удаления кэша: {e}")
                    
                    # Перепроверяем размер
                    cursor.execute('SELECT SUM(file_size) FROM tts_cache')
                    total_size_bytes = cursor.fetchone()[0] or 0
                    total_size_mb = total_size_bytes / (1024 * 1024)
                    
                    if total_size_mb <= self.max_cache_size_mb * 0.8:
                        break
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки кэша: {e}")
    
    def get_stats(self):
        """Получение статистики кэша"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute('SELECT COUNT(*) FROM tts_cache')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(file_size) FROM tts_cache')
            total_size_bytes = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(access_count) FROM tts_cache')
            total_accesses = cursor.fetchone()[0] or 0
            
            # Статистика по языкам
            cursor.execute('''
                SELECT language, COUNT(*), SUM(access_count), SUM(file_size)
                FROM tts_cache 
                GROUP BY language
            ''')
            languages_stats = cursor.fetchall()
            
            # Самые популярные фразы
            cursor.execute('''
                SELECT text, access_count, duration_sec 
                FROM tts_cache 
                ORDER BY access_count DESC 
                LIMIT 10
            ''')
            top_phrases = cursor.fetchall()
            
            # Старые записи
            cursor.execute('''
                SELECT COUNT(*) FROM tts_cache 
                WHERE created_at < ?
            ''', ((datetime.now() - timedelta(days=self.cache_ttl_days)).isoformat(),))
            old_entries = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_entries': total_entries,
                'total_size_mb': total_size_bytes / (1024 * 1024),
                'total_accesses': total_accesses,
                'languages_stats': [
                    {
                        'language': lang,
                        'count': count,
                        'accesses': accesses,
                        'size_mb': size / (1024 * 1024) if size else 0
                    }
                    for lang, count, accesses, size in languages_stats
                ],
                'top_phrases': [
                    {
                        'text': text[:100] + ('...' if len(text) > 100 else ''),
                        'access_count': access_count,
                        'duration_sec': duration_sec
                    }
                    for text, access_count, duration_sec in top_phrases
                ],
                'old_entries': old_entries,
                'max_size_mb': self.max_cache_size_mb,
                'ttl_days': self.cache_ttl_days
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики кэша: {e}")
            return {}

# Инициализируем кэш
tts_cache = TTSCache()

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
def generate_audio(text, language, speaker, sample_rate, request_id):
    """
    Генерация аудио из текста
    Возвращает путь к сгенерированному файлу
    """
    try:
        start_time = time.time()
        
        logger.info(f"\n🎵 [Request {request_id}] Начинаю генерацию аудио")
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
        
        logger.info(f"✅ [Request {request_id}] Аудио успешно сгенерировано за {generation_time:.2f} секунд")
        logger.info(f"   📁 Файл: {filename}")
        
        # Добавляем в кэш
        cache_result = tts_cache.put(text, language, speaker, sample_rate, filepath, generation_time)
        if cache_result:
            logger.info(f"✅ [Request {request_id}] Добавлено в кэш")
        
        return filepath, generation_time
        
    except Exception as e:
        logger.error(f"❌ [Request {request_id}] Ошибка генерации аудио: {str(e)}")
        raise

# ========== ФУНКЦИЯ ВЫПОЛНЕНИЯ ЗАДАЧИ В ПОТОКЕ ==========
def process_tts_request(text, language, speaker, sample_rate, request_id, callback):
    """
    Выполняет TTS запрос в отдельном потоке
    """
    try:
        logger.info(f"🧵 [Thread-{request_id}] Запуск обработки запроса")
        
        # Сначала проверяем кэш
        cache_result = tts_cache.get(text, language, speaker, sample_rate)
        
        if cache_result['hit']:
            logger.info(f"✅ [Request {request_id}] Найдено в кэше!")
            
            # Глобальная статистика
            global cache_hits
            cache_hits += 1
            
            callback({
                'success': True,
                'filepath': cache_result['file_path'],
                'request_id': request_id,
                'filename': os.path.basename(cache_result['file_path']),
                'cached': True,
                'generation_time': cache_result.get('generation_time', 0),
                'cache_hit': True
            })
            return
        
        # Если не в кэше - генерируем
        logger.info(f"🔄 [Request {request_id}] Не найдено в кэше, генерирую...")
        
        global cache_misses
        cache_misses += 1
        
        filepath, generation_time = generate_audio(text, language, speaker, sample_rate, request_id)
        
        # Вызываем колбэк с результатом
        callback({
            'success': True,
            'filepath': filepath,
            'request_id': request_id,
            'filename': os.path.basename(filepath),
            'cached': False,
            'generation_time': generation_time,
            'cache_hit': False
        })
        
    except Exception as e:
        logger.error(f"❌ [Thread-{request_id}] Ошибка обработки: {str(e)}")
        callback({
            'success': False,
            'error': str(e),
            'request_id': request_id
        })

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
            'version': '5.0',
            'status': 'running',
            'mode': 'threaded-sync-with-cache',
            'description': 'Каждый запрос обрабатывается в отдельном потоке с кэшированием',
            'cache_enabled': True,
            'endpoints': {
                '/': 'GET - главная страница',
                '/api/tts': 'POST - генерация TTS (с кэшированием)',
                '/api/health': 'GET - проверка здоровья',
                '/api/voices': 'GET - список голосов',
                '/api/cache/stats': 'GET - статистика кэша',
                '/api/cache/clear': 'POST - очистка кэша',
                '/api/test': 'GET - тестовый запрос',
                '/api/debug': 'GET - отладочная информация'
            }
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Генерация TTS - с проверкой кэша
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
        
        # Проверяем количество активных потоков
        active_count = len([t for t in threading.enumerate() 
                          if t.name.startswith('TTS-')])
        
        if active_count >= max_concurrent_threads:
            return jsonify({
                'error': 'Service is busy',
                'message': f'Too many concurrent requests ({active_count}/{max_concurrent_threads})',
                'suggestion': 'Try again in a few seconds'
            }), 429
        
        # Генерируем уникальный ID запроса
        request_id = str(uuid.uuid4())[:8]
        
        logger.info(f"\n📨 Получен TTS запрос (ID: {request_id})")
        logger.info(f"   Текст: '{req.text[:50]}...'")
        logger.info(f"   Активных потоков: {active_count}/{max_concurrent_threads}")
        
        # Используем Event для ожидания завершения потока
        done_event = threading.Event()
        result = {'done': False}
        
        def callback(response):
            result.update(response)
            result['done'] = True
            done_event.set()
        
        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(
            target=process_tts_request,
            args=(req.text, req.language, req.speaker, req.sample_rate, request_id, callback),
            name=f"TTS-{request_id}",
            daemon=True
        )
        
        thread.start()
        
        # Ждем завершения потока (синхронное ожидание)
        logger.info(f"⏳ Ожидание завершения генерации (ID: {request_id})...")
        
        # Таймаут: 60 секунд на генерацию
        if not done_event.wait(timeout=60):
            logger.error(f"❌ [Request {request_id}] Таймаут генерации")
            return jsonify({
                'error': 'Generation timeout',
                'request_id': request_id,
                'message': 'Generation took too long'
            }), 504
        
        # Проверяем результат
        if not result['success']:
            logger.error(f"❌ [Request {request_id}] Ошибка в результате: {result.get('error')}")
            return jsonify({
                'error': result.get('error', 'Unknown error'),
                'request_id': request_id
            }), 500
        
        # Получаем путь к файлу
        filepath = result['filepath']
        filename = result['filename']
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File was not created'}), 500
        
        logger.info(f"📤 [Request {request_id}] Отправляю файл: {filename}")
        
        # Статистика кэша для ответа
        cache_stats = f"Cache: {'HIT' if result.get('cache_hit') else 'MISS'}"
        if not result.get('cache_hit'):
            cache_stats += f", Generation time: {result.get('generation_time', 0):.2f}s"
        
        logger.info(f"📊 [Request {request_id}] {cache_stats}")
        
        # Отправляем файл
        response = send_file(
            filepath,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
        # Добавляем заголовки с информацией о кэше
        response.headers['X-Cache-Hit'] = 'true' if result.get('cache_hit') else 'false'
        response.headers['X-Generation-Time'] = f"{result.get('generation_time', 0):.2f}"
        response.headers['X-Cache-Stats'] = f"Hits: {cache_hits}, Misses: {cache_misses}"
        
        # Очистка файла после отправки (только для временных файлов, не кэшированных)
        if not result.get('cached'):
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
        logger.error(f"❌ Ошибка в tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        
        # Считаем активные потоки TTS
        tts_threads = [t for t in threading.enumerate() if t.name.startswith('TTS-')]
        
        # Получаем статистику кэша
        cache_stats = tts_cache.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'service': 'zindaki-tts-service',
            'version': '5.0',
            'mode': 'threaded-sync-with-cache',
            'description': 'Синхронная генерация с кэшированием',
            'active_threads': threading.active_count(),
            'active_tts_threads': len(tts_threads),
            'max_concurrent_threads': max_concurrent_threads,
            'models_loaded': list(tts_models.keys()),
            'models_count': len(tts_models),
            'cache_stats': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_ratio': cache_hits / max((cache_hits + cache_misses), 1),
                'total_entries': cache_stats.get('total_entries', 0),
                'total_size_mb': cache_stats.get('total_size_mb', 0)
            },
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

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Получение подробной статистики кэша"""
    try:
        stats = tts_cache.get_stats()
        
        return jsonify({
            'cache_enabled': True,
            'global_stats': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_ratio': cache_hits / max((cache_hits + cache_misses), 1),
                'total_requests': cache_hits + cache_misses
            },
            'cache_details': stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики кэша: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Очистка кэша"""
    try:
        data = request.get_json() or {}
        clear_all = data.get('clear_all', False)
        days_old = data.get('days_old', 7)
        
        conn = sqlite3.connect(tts_cache.db_path)
        cursor = conn.cursor()
        
        if clear_all:
            # Удаляем все записи
            cursor.execute('SELECT cache_key, file_path FROM tts_cache')
            records = cursor.fetchall()
            
            for cache_key, file_path in records:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
            
            cursor.execute('DELETE FROM tts_cache')
            message = "Полная очистка кэша"
            
        else:
            # Удаляем записи старше указанного количества дней
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cursor.execute('SELECT cache_key, file_path FROM tts_cache WHERE created_at < ?', 
                          (cutoff_date.isoformat(),))
            records = cursor.fetchall()
            
            for cache_key, file_path in records:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
            
            cursor.execute('DELETE FROM tts_cache WHERE created_at < ?', 
                          (cutoff_date.isoformat(),))
            
            message = f"Очистка кэша старше {days_old} дней"
        
        deleted_count = conn.total_changes
        conn.commit()
        conn.close()
        
        # Сбрасываем статистику хитов/миссов
        global cache_hits, cache_misses
        cache_hits = 0
        cache_misses = 0
        
        logger.info(f"🗑️ {message}: удалено {deleted_count} записей")
        
        return jsonify({
            'success': True,
            'message': message,
            'deleted_count': deleted_count,
            'clear_all': clear_all,
            'days_old': days_old if not clear_all else None,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка очистки кэша: {e}")
        return jsonify({'error': str(e)}), 500

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
            'tts_cache_enabled': True,
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
    
    # Проверяем директорию кэша
    cache_files = []
    cache_dir = '/app/tts_cache/audio'
    if os.path.exists(cache_dir):
        cache_files = os.listdir(cache_dir)
    
    # Получаем информацию о потоках
    thread_info = []
    tts_threads = []
    for thread in threading.enumerate():
        if thread.name.startswith('TTS-'):
            tts_threads.append(thread.name)
        thread_info.append({
            'name': thread.name,
            'daemon': thread.daemon,
            'alive': thread.is_alive()
        })
    
    # Статистика кэша
    cache_stats = tts_cache.get_stats()
    
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version.split()[0],
        'templates_dir': templates_dir,
        'template_files': template_files,
        'temp_audio_dir': temp_dir,
        'temp_files_count': len(temp_files),
        'tts_cache_dir': cache_dir,
        'tts_cache_files_count': len(cache_files),
        'tts_cache_files': cache_files[:5],
        'models_loaded': list(tts_models.keys()),
        'active_threads': threading.active_count(),
        'active_tts_threads': len(tts_threads),
        'tts_thread_names': tts_threads[:10],
        'max_concurrent_threads': max_concurrent_threads,
        'cache_stats': {
            'global_hits': cache_hits,
            'global_misses': cache_misses,
            'hit_ratio': cache_hits / max((cache_hits + cache_misses), 1),
            'cache_details': cache_stats
        },
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
    """Периодическая очистка временных файлов и проверка кэша"""
    while True:
        time.sleep(3600)  # Каждый час
        
        # Очистка файлов
        cleanup_temp_files()
        
        # Проверяем кэш
        try:
            stats = tts_cache.get_stats()
            logger.info(f"📊 Статистика кэша: {stats.get('total_entries', 0)} записей, "
                       f"{stats.get('total_size_mb', 0):.1f} MB")
        except:
            pass

# Регистрируем очистку при завершении
atexit.register(cleanup_temp_files)

# ========== ЗАПУСК СЕРВИСА ==========

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🎵 ZINDAKI TTS SERVICE - Версия с кэшированием v5.0")
    print("=" * 70)
    print(f"📅 Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python версия: {sys.version.split()[0]}")
    print(f"🔥 PyTorch версия: {torch.__version__}")
    print(f"🎵 TorchAudio версия: {torchaudio.__version__}")
    print(f"📁 Кэш директория: {os.environ.get('TORCH_HOME')}")
    print(f"📁 Директория временных файлов: /app/temp_audio")
    print(f"📁 Директория TTS кэша: /app/tts_cache")
    print(f"🧵 Максимальное количество потоков: {max_concurrent_threads}")
    
    # Статистика кэша при старте
    cache_stats = tts_cache.get_stats()
    print(f"📊 Кэш при старте: {cache_stats.get('total_entries', 0)} записей, "
          f"{cache_stats.get('total_size_mb', 0):.1f} MB")
    
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
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True, name="Cache-Cleanup")
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
    print("   POST /api/tts           - Генерация TTS (с кэшированием)")
    print("   GET  /api/health        - Проверка здоровья сервиса")
    print("   GET  /api/cache/stats   - Статистика кэша")
    print("   POST /api/cache/clear   - Очистка кэша")
    print("   GET  /api/voices        - Список доступных голосов")
    print("=" * 70)
    print("\n📝 Режим работы: Синхронная генерация с кэшированием")
    print("   • Каждый запрос проверяется в кэше")
    print("   • При отсутствии в кэше - генерация в отдельном потоке")
    print("   • Результаты автоматически добавляются в кэш")
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )