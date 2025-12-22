import os
import uuid
import torch
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from rq import Queue
from rq.job import Job
from worker import conn
import torchaudio
import tempfile

# Модели Silero TTS (легковесные и быстрые)
MODELS = {
    'en': 'v3_en.pt',
    'ru': 'v3_ru.pt'
}

app = Flask(__name__)
CORS(app)  # Разрешаем запросы с фронтенда
q = Queue(connection=conn)  # Очередь задач

# Модели загружаются один раз при старте воркера, а не здесь.
# Этот код выполняется внутри фоновой задачи (worker.py).

class TTSRequest(BaseModel):
    """Модель для валидации входящих запросов"""
    text: str
    language: str = 'ru'  # По умолчанию русский
    speaker: str = 'aidar'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
    sample_rate: int = 48000

def generate_audio_in_background(request_data: dict):
    """Фоновая задача для генерации аудио.
    Выполняется воркером отдельно от основного потока Flask."""
    try:
        req = TTSRequest(**request_data)
        lang = req.language
        model_id = MODELS.get(lang)
        
        if not model_id:
            return {'error': f'Unsupported language: {lang}'}
        
        # Загружаем модель и утилиты Silero
        device = torch.device('cpu')  # Можно использовать 'cuda', если есть GPU
        model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=lang,
            speaker=req.speaker
        )
        model.to(device)
        
        # Генерация аудио
        audio = model.apply_tts(
            text=req.text,
            speaker=req.speaker,
            sample_rate=req.sample_rate
        )
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio.unsqueeze(0), req.sample_rate)
            return tmp_file.name
            
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """Основной endpoint для запроса озвучки"""
    try:
        data = request.get_json()
        req = TTSRequest(**data)
        
        # Помещаем задачу в очередь и сразу возвращаем ID job
        job = q.enqueue(generate_audio_in_background, request_data=data)
        return jsonify({
            'job_id': job.get_id(),
            'status': 'queued',
            'message': 'Задача поставлена в очередь на обработку'
        }), 202  # 202 Accepted
        
    except ValidationError as e:
        return jsonify({'error': 'Некорректные данные', 'details': e.errors()}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=conn)
        
        if job.is_finished:
            result = job.result
            if isinstance(result, dict) and 'error' in result:
                return jsonify(result), 500
            # Возвращаем готовый аудиофайл
            return send_file(result, mimetype='audio/wav', as_attachment=True, download_name=f'{job_id}.wav')
        elif job.is_failed:
            return jsonify({'error': 'Задача завершилась с ошибкой', 'status': 'failed'}), 500
        else:
            return jsonify({'status': job.get_status()}), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов для каждого языка"""
    voices = {
        'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'random'],
        'en': ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'random']
    }
    return jsonify(voices)

if __name__ == '__main__':
    # Для продакшена используйте gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)
