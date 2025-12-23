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

# Инициализация Redis
redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'tts-redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=1,
    socket_connect_timeout=5,
    socket_timeout=5
)
q = Queue(connection=redis_conn, default_timeout=600)

# Модели Silero TTS
MODELS = {
    'en': 'v3_en',
    'ru': 'v3_ru'
}

app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Глобальные переменные для хранения загруженных моделей
tts_models = {}

class TTSRequest(BaseModel):
    """Модель для валидации входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'  # Женский голос по умолчанию
    sample_rate: int = 24000
    put_accent: bool = True
    put_yo: bool = True

def load_model(language, speaker):
    """Загружаем модель Silero TTS (кешируем в памяти)"""
    model_key = f"{language}_{speaker}"
    
    if model_key not in tts_models:
        print(f"Loading model: {language}, speaker: {speaker}")
        
        device = torch.device('cpu')
        
        try:
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=speaker
            )
            model.to(device)
            
            tts_models[model_key] = {
                'model': model,
                'example_text': example_text,
                'device': device
            }
            print(f"✅ Model {model_key} loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model {model_key}: {e}")
            raise
    
    return tts_models[model_key]

def generate_audio(text, language, speaker, sample_rate, put_accent=True, put_yo=True):
    """Генерация аудио"""
    try:
        print(f"Generating audio for: '{text[:100]}...' (lang: {language}, speaker: {speaker})")
        
        model_info = load_model(language, speaker)
        model = model_info['model']
        device = model_info['device']
        
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo
        )
        
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            delete=False,
            dir=temp_dir
        )
        
        torchaudio.save(
            temp_file.name, 
            audio.unsqueeze(0), 
            sample_rate,
            format='wav'
        )
        
        print(f"✅ Audio generated: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"❌ Error in generate_audio: {str(e)}")
        raise

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
        
        if req.language not in MODELS:
            return jsonify({'error': f'Unsupported language: {req.language}'}), 400
        
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
            'estimated_time': '10-30 seconds'
        }), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Invalid data', 'details': e.errors()}), 400
    except Exception as e:
        print(f"Error in tts_request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Проверка статуса задачи"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({'error': 'Job result is empty'}), 500
            
            if isinstance(result, str) and os.path.exists(result):
                try:
                    response = send_file(
                        result,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=f'tts_{job_id}.wav'
                    )
                    
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
                'position': job.get_position() if hasattr(job, 'get_position') else 'unknown'
            }), 200
            
    except Exception as e:
        print(f"Error in get_status: {str(e)}")
        return jsonify({'error': f'Job not found: {str(e)}'}), 404

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """Возвращает список доступных голосов"""
    voices = {
        'ru': ['baya', 'kseniya', 'xenia'],  # Только женские
        'en': ['en_1', 'en_3']  # Женские английские
    }
    return jsonify(voices)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    try:
        redis_conn.ping()
        models_loaded = len(tts_models) > 0
        
        return jsonify({
            'status': 'healthy',
            'redis': 'connected',
            'models_loaded': models_loaded,
            'queue_size': len(q),
            'service': 'zindaki-tts',
            'female_voices_loaded': list(tts_models.keys())
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def cleanup_temp_files():
    """Очистка временных файлов"""
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_audio')
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

atexit.register(cleanup_temp_files)

if __name__ == '__main__':
    os.makedirs('temp_audio', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )