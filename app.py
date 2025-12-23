#!/usr/bin/env python3
"""
ZINDAKI TTS SERVICE - Полная исправленная версия
С исправлением проблемы создания аудио файлов
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

# ========== КОРРЕКТНЫЕ ИМЕНА ДИКТОРОВ SILERO ==========
SPEAKER_MAPPING = {
    'ru': {
        'baya': 'baya_16khz',
        'kseniya': 'kseniya_16khz',
        'aidar': 'aidar_16khz',
        'irina': 'irina_16khz',
        'natasha': 'natasha_16khz',
        'ruslan': 'ruslan_16khz',
        'xenia': 'xenia_16khz',
        'random': 'random'
    },
    'en': {
        'en_0': 'en_0',           # Amy (female)
        'en_1': 'en_1',           # Eric (male)
        'en_2': 'en_2',           # Kyle (male)
        'en_3': 'en_3',           # Lucy (female)
        'en_4': 'en_4',           # Nancy (female)
        'en_5': 'en_5',           # Grace (female)
        'en_6': 'en_6',           # Anthony (male)
        'en_7': 'en_7',           # Emma (female)
        'en_8': 'en_8',           # Brian (male)
        'en_9': 'en_9',           # Jenny (female)
        'en_10': 'en_10',         # Chris (male)
        'en_11': 'en_11',         # Elizabeth (female)
        'en_12': 'en_12',         # George (male)
        'en_13': 'en_13',         # Linda (female)
        'en_14': 'en_14',         # Mark (male)
        'en_15': 'en_15',         # Mary (female)
        'en_16': 'en_16',         # Paul (male)
        'en_17': 'en_17',         # Sarah (female)
        'en_18': 'en_18',         # Steven (male)
        'en_19': 'en_19',         # Susan (female)
        'en_20': 'en_20',         # Thomas (male)
        'en_21': 'en_21',         # Victoria (female)
        'en_22': 'en_22',         # William (male)
        'en_23': 'en_23',         # Heather (female)
        'en_24': 'en_24',         # James (male)
        'en_25': 'en_25',         # Jennifer (female)
        'en_26': 'en_26',         # John (male)
        'en_27': 'en_27',         # Karen (female)
        'en_28': 'en_28',         # Michael (male)
        'en_29': 'en_29',         # Michelle (female)
        'en_30': 'en_30',         # Richard (male)
        'en_31': 'en_31',         # Sandra (female)
        'en_32': 'en_32',         # David (male)
        'en_33': 'en_33',         # Barbara (female)
        'en_34': 'en_34',         # Charles (male)
        'en_35': 'en_35',         # Margaret (female)
        'en_36': 'en_36',         # Joseph (male)
        'en_37': 'en_37',         # Dorothy (female)
        'en_38': 'en_38',         # Christopher (male)
        'en_39': 'en_39',         # Lisa (female)
        'en_40': 'en_40',         # Daniel (male)
        'en_41': 'en_41',         # Nancy (female) - duplicate name
        'en_42': 'en_42',         # Matthew (male)
        'en_43': 'en_43',         # Betty (female)
        'en_44': 'en_44',         # Anthony (male) - duplicate name
        'en_45': 'en_45',         # Helen (female)
        'en_46': 'en_46',         # Donald (male)
        'en_47': 'en_47',         # Carol (female)
        'en_48': 'en_48',         # Ronald (male)
        'en_49': 'en_49',         # Ruth (female)
        'en_50': 'en_50',         # Kevin (male)
        'en_51': 'en_51',         # Shirley (female)
        'en_52': 'en_52',         # Edward (male)
        'en_53': 'en_53',         # Deborah (female)
        'en_54': 'en_54',         # Jason (male)
        'en_55': 'en_55',         # Jessica (female)
        'en_56': 'en_56',         # Ryan (male)
        'en_57': 'en_57',         # Cynthia (female)
        'en_58': 'en_58',         # Jacob (male)
        'en_59': 'en_59',         # Angela (female)
        'en_60': 'en_60',         # Gary (male)
        'en_61': 'en_61',         # Melissa (female)
        'en_62': 'en_62',         # Nicholas (male)
        'en_63': 'en_63',         # Brenda (female)
        'en_64': 'en_64',         # Eric (male) - duplicate name
        'en_65': 'en_65',         # Pamela (female)
        'en_66': 'en_66',         # Jonathan (male)
        'en_67': 'en_67',         # Rebecca (female)
        'en_68': 'en_68',         # Stephen (male)
        'en_69': 'en_69',         # Sharon (female)
        'en_70': 'en_70',         # Larry (male)
        'en_71': 'en_71',         # Kathleen (female)
        'en_72': 'en_72',         # Justin (male)
        'en_73': 'en_73',         # Amy (female) - duplicate name
        'en_74': 'en_74',         # Brandon (male)
        'en_75': 'en_75',         # Anna (female)
        'en_76': 'en_76',         # Gregory (male)
        'en_77': 'en_77',         # Ruth (female) - duplicate name
        'en_78': 'en_78',         # Samuel (male)
        'en_79': 'en_79',         # Virginia (female)
        'en_80': 'en_80',         # Frank (male)
        'en_81': 'en_81',         # Katherine (female)
        'en_82': 'en_82',         # Raymond (male)
        'en_83': 'en_83',         # Christine (female)
        'en_84': 'en_84',         # Patrick (male)
        'en_85': 'en_85',         # Janet (female)
        'en_86': 'en_86',         # Jack (male)
        'en_87': 'en_87',         # Carolyn (female)
        'en_88': 'en_88',         # Dennis (male)
        'en_89': 'en_89',         # Catherine (female)
        'en_90': 'en_90',         # Jerry (male)
        'en_91': 'en_91',         # Frances (female)
        'en_92': 'en_92',         # Tyler (male)
        'en_93': 'en_93',         # Ann (female)
        'en_94': 'en_94',         # Aaron (male)
        'en_95': 'en_95',         # Joyce (female)
        'en_96': 'en_96',         # Jose (male)
        'en_97': 'en_97',         # Diane (female)
        'en_98': 'en_98',         # Henry (male)
        'en_99': 'en_99',         # Alice (female)
        'en_100': 'en_100',       # Adam (male)
        'en_101': 'en_101',       # Julia (female)
        'en_102': 'en_102',       # Willie (male)
        'en_103': 'en_103',       # Judy (female)
        'en_104': 'en_104',       # Nathan (male)
        'en_105': 'en_105',       # Marie (female)
        'en_106': 'en_106',       # Evan (male)
        'en_107': 'en_107',       # Beverly (female)
        'en_108': 'en_108',       # Christian (male)
        'en_109': 'en_109',       # Denise (female)
        'en_110': 'en_110',       # Austin (male)
        'en_111': 'en_111',       # Marilyn (female)
        'en_112': 'en_112',       # Billy (male)
        'en_113': 'en_113',       # Amber (female)
        'en_114': 'en_114',       # Bruce (male)
        'en_115': 'en_115',       # Madison (female)
        'en_116': 'en_116',       # Bryan (male)
        'en_117': 'en_117',       # Danielle (female)
        'en_118': 'en_118',       # Albert (male)
        'en_119': 'en_119',       # Rose (female)
        'en_120': 'en_120',       # Lawrence (male)
        'en_121': 'en_121',       # Sophia (female)
        'en_122': 'en_122',       # Dylan (male)
        'en_123': 'en_123',       # Olivia (female)
        'en_124': 'en_124',       # Zachary (male)
        'en_125': 'en_125',       # Grace (female) - duplicate name
        'en_126': 'en_126',       # Kyle (male) - duplicate name
        'en_127': 'en_127',       # Chloe (female)
        'en_128': 'en_128',       # Louis (male)
        'en_129': 'en_129',       # Hailey (female)
        'en_130': 'en_130',       # Wayne (male)
        'en_131': 'en_131',       # Gabriella (female)
        'en_132': 'en_132',       # Ethan (male)
        'en_133': 'en_133',       # Allison (female)
        'en_134': 'en_134',       # Randy (male)
        'en_135': 'en_135',       # Lily (female)
        'en_136': 'en_136',       # Philip (male)
        'en_137': 'en_137',       # Megan (female)
        'en_138': 'en_138',       # Harry (male)
        'en_139': 'en_139',       # Savannah (female)
        'en_140': 'en_140',       # Vincent (male)
        'en_141': 'en_141',       # Alyssa (female)
        'en_142': 'en_142',       # Noah (male)
        'en_143': 'en_143',       # Haley (female)
        'en_144': 'en_144',       # Shawn (male)
        'en_145': 'en_145',       # Stella (female)
        'en_146': 'en_146',       # Connor (male)
        'en_147': 'en_147',       # Penelope (female)
        'en_148': 'en_148',       # Carlos (male)
        'en_149': 'en_149',       # Natalie (female)
    },
    'de': {
        'eva_k': 'eva_k',
        'karlsson': 'karlsson'
    },
    'es': {
        'es_0': 'es_0',
        'es_1': 'es_1'
    },
    'fr': {
        'fr_0': 'fr_0',
        'fr_1': 'fr_1'
    }
}

# ========== МОДЕЛЬ ЗАПРОСА ==========
class TTSRequest(BaseModel):
    """Валидация входящих запросов"""
    text: str
    language: str = 'ru'
    speaker: str = 'baya'
    sample_rate: int = 16000
    speed: float = 1.0
    
    class Config:
        extra = 'forbid'

# ========== ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ==========
def load_tts_model(language='ru', user_speaker='baya'):
    """
    Загружает модель Silero TTS по требованию
    """
    # Формируем ключ для кэша
    model_key = f"{language}_{user_speaker}"
    
    if model_key not in tts_models:
        print(f"📥 Загружаю модель TTS: {language}/{user_speaker}")
        
        # Получаем правильное имя диктора
        if language in SPEAKER_MAPPING and user_speaker in SPEAKER_MAPPING[language]:
            correct_speaker = SPEAKER_MAPPING[language][user_speaker]
        else:
            # Значения по умолчанию
            if language == 'ru':
                correct_speaker = 'baya_16khz'
            else:
                correct_speaker = user_speaker
        
        print(f"   Использую правильное имя: {correct_speaker}")
        
        # Устанавливаем директорию кэша
        torch.hub.set_dir('/app/cache/torch/hub')
        
        try:
            # Загружаем модель
            result = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=correct_speaker,
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            
            print(f"✅ Модель загружена ({len(result)} элементов)")
            
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
            
            print(f"   Sample rate: {result[2]} Hz")
            print(f"   Пример текста: {result[3][:50]}...")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    return tts_models[model_key]

# ========== ФУНКЦИЯ ГЕНЕРАЦИИ АУДИО ==========
def generate_audio(text, language, speaker, sample_rate, speed=1.0):
    """
    Генерация аудио из текста
    Исправленная версия для правильного создания WAV файлов
    """
    try:
        start_time = time.time()
        
        print(f"\n🎵 Начинаю генерацию аудио")
        print(f"   Язык: {language}, Голос: {speaker}")
        print(f"   Текст: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"   Длина: {len(text)} символов")
        print(f"   Скорость: {speed}x")
        
        # Загружаем или получаем модель из кэша
        model_info = load_tts_model(language, speaker)
        
        # Получаем компоненты модели
        model = model_info['model']
        symbols = model_info['symbols']
        target_sample_rate = model_info['sample_rate']
        apply_tts_func = model_info['apply_tts']
        device = model_info['device']
        
        print(f"   🔊 Использую голос: {model_info['correct_speaker']}")
        print(f"   🎚️  Частота: {target_sample_rate} Hz")
        
        # Генерация аудио
        audio_result = apply_tts_func(
            texts=[text],           # Список текстов
            model=model,
            sample_rate=target_sample_rate,
            symbols=symbols,
            device=device
        )
        
        # ДИАГНОСТИКА
        print(f"   📊 Тип результата: {type(audio_result)}")
        
        # Преобразование результата в тензор
        audio_tensor = None
        
        # 1. Если результат - список
        if isinstance(audio_result, list):
            print(f"   📊 Длина списка: {len(audio_result)}")
            if len(audio_result) == 0:
                raise ValueError("apply_tts вернул пустой список")
            
            first_element = audio_result[0]
            print(f"   📊 Тип первого элемента: {type(first_element)}")
            
            if isinstance(first_element, torch.Tensor):
                audio_tensor = first_element
                print(f"   ✅ Использую тензор из списка")
            elif hasattr(first_element, 'numpy'):
                audio_tensor = torch.from_numpy(first_element.numpy())
                print(f"   ✅ Конвертирован из numpy")
            else:
                try:
                    audio_tensor = torch.tensor(first_element, dtype=torch.float32)
                    print(f"   ✅ Создан тензор из данных")
                except:
                    raise ValueError(f"Не могу преобразовать элемент в тензор: {type(first_element)}")
        
        # 2. Если результат напрямую тензор
        elif isinstance(audio_result, torch.Tensor):
            audio_tensor = audio_result
            print(f"   ✅ Результат - напрямую тензор")
        
        # 3. Если это numpy array
        elif hasattr(audio_result, '__array__'):
            import numpy as np
            audio_tensor = torch.from_numpy(np.array(audio_result, dtype=np.float32))
            print(f"   ✅ Конвертирован из numpy array")
        
        else:
            raise ValueError(f"Неизвестный тип результата: {type(audio_result)}")
        
        # Проверяем, что получили тензор
        if audio_tensor is None:
            raise ValueError("Не удалось получить аудио тензор")
        
        # ПРОВЕРКА И НОРМАЛИЗАЦИЯ ТЕНЗОРА
        print(f"   🔧 Подготовка тензора...")
        print(f"   📐 Исходный shape: {audio_tensor.shape}")
        print(f"   🎛️  Исходный dtype: {audio_tensor.dtype}")
        
        # Конвертируем в float32 если нужно
        if audio_tensor.dtype != torch.float32:
            print(f"   🔄 Конвертирую в float32")
            audio_tensor = audio_tensor.to(torch.float32)
        
        # Нормализация значений
        max_val = torch.max(torch.abs(audio_tensor))
        print(f"   📊 Максимальное абсолютное значение: {max_val.item()}")
        
        if max_val > 1.0:
            print(f"   🔄 Нормализую значения (делю на {max_val.item():.3f})")
            audio_tensor = audio_tensor / max_val
        elif max_val == 0:
            print(f"   ⚠️  Внимание: все значения равны нулю")
        
        # Приводим к правильной размерности: (каналы, время)
        print(f"   🔧 Проверяю размерность...")
        
        if audio_tensor.ndim == 1:
            # (время) -> (1, время) - моно звук
            audio_tensor = audio_tensor.unsqueeze(0)
            print(f"   🔄 Преобразование: 1D -> 2D, новый shape: {audio_tensor.shape}")
        elif audio_tensor.ndim == 2:
            # Проверяем ориентацию
            if audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[0] > 10:
                # Возможно (время, каналы) -> (каналы, время)
                audio_tensor = audio_tensor.transpose(0, 1)
                print(f"   🔄 Транспонирование: новый shape: {audio_tensor.shape}")
        elif audio_tensor.ndim == 3:
            # (batch, каналы, время) -> (каналы, время)
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
                print(f"   🔄 Удален batch dimension: новый shape: {audio_tensor.shape}")
            else:
                raise ValueError(f"Неожиданная 3D размерность: {audio_tensor.shape}")
        else:
            raise ValueError(f"Неожиданная размерность тензора: {audio_tensor.ndim}")
        
        print(f"   📐 Финальный shape: {audio_tensor.shape}")
        
        # Применяем изменение скорости если нужно
        if speed != 1.0:
            try:
                from torchaudio.transforms import Speed
                print(f"   🏎️  Применяю изменение скорости: {speed}x")
                speed_transform = Speed(
                    orig_freq=target_sample_rate,
                    factor=speed
                )
                audio_tensor = speed_transform(audio_tensor)
                print(f"   📐 Shape после изменения скорости: {audio_tensor.shape}")
            except ImportError:
                print(f"   ⚠️  torchaudio.transforms недоступен, пропускаю изменение скорости")
        
        # Создаем временный файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"tts_{language}_{speaker}_{timestamp}.wav"
        filepath = os.path.join(temp_dir, filename)
        
        # Сохраняем аудио в файл
        print(f"   💾 Сохраняю аудио в файл: {filepath}")
        
        try:
            # Способ 1: Используем torchaudio.save
            torchaudio.save(
                filepath,
                audio_tensor,
                target_sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            print(f"   ✅ Сохранено через torchaudio")
            
        except Exception as e1:
            print(f"   ⚠️  Ошибка torchaudio.save: {e1}")
            
            try:
                # Способ 2: Используем wave модуль напрямую
                import wave
                import numpy as np
                
                # Конвертируем в numpy
                audio_np = audio_tensor.squeeze().numpy()
                
                # Нормализация для 16-bit PCM
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)
                
                # Масштабирование до 16-bit PCM
                int16_max = 32767
                audio_np = (audio_np * int16_max).astype(np.int16)
                
                # Сохраняем через wave
                with wave.open(filepath, 'w') as wav_file:
                    wav_file.setnchannels(1)  # моно
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(target_sample_rate)
                    wav_file.writeframes(audio_np.tobytes())
                
                print(f"   ✅ Сохранено через wave модуль")
                
            except Exception as e2:
                print(f"   ⚠️  Ошибка wave.save: {e2}")
                
                try:
                    # Способ 3: Используем scipy если доступен
                    import scipy.io.wavfile
                    
                    audio_np = audio_tensor.squeeze().numpy()
                    int16_max = 32767
                    audio_np = (audio_np * int16_max).astype(np.int16)
                    
                    scipy.io.wavfile.write(filepath, target_sample_rate, audio_np)
                    print(f"   ✅ Сохранено через scipy")
                    
                except ImportError:
                    print(f"   ❌ scipy не установлен")
                    raise ValueError(f"Не удалось сохранить файл никаким методом")
                except Exception as e3:
                    print(f"   ❌ Ошибка scipy.save: {e3}")
                    raise
        
        # Проверяем, что файл создан
        if os.path.exists(filepath):
            filesize = os.path.getsize(filepath) / 1024
            print(f"   ✅ Файл создан успешно!")
            print(f"   📁 Размер файла: {filesize:.1f} KB")
        else:
            raise ValueError(f"Файл не был создан: {filepath}")
        
        # Вычисляем статистику
        generation_time = time.time() - start_time
        audio_duration = audio_tensor.shape[-1] / target_sample_rate
        
        print(f"\n✅ Аудио успешно сгенерировано!")
        print(f"   ⏱️  Время генерации: {generation_time:.2f} секунд")
        print(f"   🕒 Длительность аудио: {audio_duration:.2f} секунд")
        print(f"   📁 Файл: {filepath}")
        print(f"   📊 Размер: {filesize:.1f} KB")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Ошибка генерации аудио: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ========== API МАРШРУТЫ ==========

@app.route('/')
def index():
    """Главная страница с веб-интерфейсом"""
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({
            'service': 'Zindaki TTS Service',
            'version': '2.0',
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
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/tts', methods=['POST'])
def tts_request():
    """
    Основной endpoint для генерации TTS
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
        
        if len(req.text) > 10000:
            return jsonify({
                'error': f'Text too long ({len(req.text)} chars). Max is 10000.'
            }), 400
        
        print(f"\n📨 Получен TTS запрос:")
        print(f"   🌐 Язык: {req.language}")
        print(f"   🗣️  Голос: {req.speaker}")
        print(f"   📝 Длина текста: {len(req.text)} символов")
        print(f"   🏎️  Скорость: {req.speed}x")
        
        # Создаем фоновую задачу
        job = queue.enqueue(
            generate_audio,
            args=(req.text, req.language, req.speaker, req.sample_rate, req.speed),
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
    """Проверка статуса задачи и получение результата"""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            
            if not result or not os.path.exists(result):
                return jsonify({'error': 'No audio file generated'}), 500
            
            # Проверяем расширение файла
            if not result.endswith('.wav'):
                print(f"⚠️  Файл имеет неправильное расширение: {result}")
                # Переименовываем если это .wav.json
                if result.endswith('.wav.json'):
                    new_result = result.replace('.wav.json', '.wav')
                    if os.path.exists(new_result):
                        result = new_result
                    else:
                        # Пробуем найти WAV файл
                        base_name = os.path.splitext(result)[0]
                        wav_file = base_name + '.wav'
                        if os.path.exists(wav_file):
                            result = wav_file
                        else:
                            raise ValueError(f"WAV файл не найден для {result}")
            
            # Отправляем аудио файл
            response = send_file(
                result,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f'tts_{job_id}.wav'
            )
            
            # Очистка файла после отправки
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
            # Задача еще выполняется
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
    """Проверка здоровья сервиса"""
    try:
        # Проверка Redis
        redis_conn.ping()
        
        # Пробуем загрузить модель, если еще не загружена
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
            'torchaudio_version': torchaudio.__version__,
            'cuda_available': torch.cuda.is_available(),
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
    # Основные голоса
    voices_info = {
        'ru': [
            {'id': 'baya', 'name': 'Байя', 'gender': 'female', 'description': 'Чистый женский голос'},
            {'id': 'kseniya', 'name': 'Ксения', 'gender': 'female', 'description': 'Мягкий женский голос'},
            {'id': 'aidar', 'name': 'Айдар', 'gender': 'male', 'description': 'Мужской голос'},
            {'id': 'irina', 'name': 'Ирина', 'gender': 'female', 'description': 'Женский голос'},
            {'id': 'natasha', 'name': 'Наташа', 'gender': 'female', 'description': 'Женский голос'},
            {'id': 'ruslan', 'name': 'Руслан', 'gender': 'male', 'description': 'Мужской голос'},
            {'id': 'xenia', 'name': 'Ксения 2', 'gender': 'female', 'description': 'Альтернативный женский голос'},
            {'id': 'random', 'name': 'Случайный', 'gender': 'random', 'description': 'Случайный голос'}
        ],
        'en': [
            {'id': 'en_0', 'name': 'Amy', 'gender': 'female', 'description': 'English female voice'},
            {'id': 'en_1', 'name': 'Eric', 'gender': 'male', 'description': 'English male voice'},
            {'id': 'en_2', 'name': 'Kyle', 'gender': 'male', 'description': 'English male voice'},
            {'id': 'en_3', 'name': 'Lucy', 'gender': 'female', 'description': 'English female voice'},
            {'id': 'en_4', 'name': 'Nancy', 'gender': 'female', 'description': 'English female voice'},
            {'id': 'en_5', 'name': 'Grace', 'gender': 'female', 'description': 'English female voice'},
        ]
    }
    
    return jsonify({
        'all_voices': voices_info,
        'loaded_voices': list(tts_models.keys()),
        'total_loaded': len(tts_models),
        'speaker_mapping': {k: list(v.keys())[:5] for k, v in SPEAKER_MAPPING.items()},
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint для проверки работы сервиса"""
    try:
        # Загружаем тестовую модель
        model_info = load_tts_model('ru', 'baya')
        
        # Тестовая генерация
        test_text = "Привет! Это тестовое сообщение TTS сервиса. Тестирование работы генерации аудио."
        
        print(f"🧪 Тестовый запрос: {test_text}")
        
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
            if isinstance(audio, torch.Tensor):
                audio_shape = str(audio.shape)
                audio_dtype = str(audio.dtype)
            else:
                audio_shape = f"type: {type(audio)}"
                audio_dtype = "N/A"
        else:
            audio = audio_result
            result_type = str(type(audio_result))
            audio_shape = str(audio.shape) if hasattr(audio, 'shape') else 'no shape'
            audio_dtype = str(audio.dtype) if hasattr(audio, 'dtype') else 'no dtype'
        
        # Пробуем сохранить тестовый файл
        temp_dir = '/app/temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        test_file = os.path.join(temp_dir, 'test_output.wav')
        
        if isinstance(audio, torch.Tensor):
            torchaudio.save(test_file, audio.unsqueeze(0) if audio.ndim == 1 else audio, 
                          model_info['sample_rate'])
            file_exists = os.path.exists(test_file)
            if file_exists:
                os.remove(test_file)
        else:
            file_exists = False
        
        return jsonify({
            'success': True,
            'message': 'TTS сервис работает корректно',
            'result_type': result_type,
            'audio_shape': audio_shape,
            'audio_dtype': audio_dtype,
            'sample_rate': model_info['sample_rate'],
            'model_loaded': True,
            'correct_speaker': model_info['correct_speaker'],
            'models_in_cache': list(tts_models.keys()),
            'test_file_saved': file_exists,
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
    """Отладочная информация"""
    import numpy as np
    
    # Проверяем наличие директорий
    templates_dir = '/app/templates'
    temp_dir = '/app/temp_audio'
    
    template_files = []
    if os.path.exists(templates_dir):
        template_files = os.listdir(templates_dir)
    
    temp_files = []
    if os.path.exists(temp_dir):
        temp_files = os.listdir(temp_dir)
    
    return jsonify({
        'torch_version': torch.__version__,
        'torchaudio_version': torchaudio.__version__,
        'python_version': sys.version,
        'environment': {
            'TORCH_HOME': os.environ.get('TORCH_HOME'),
            'HF_HOME': os.environ.get('HF_HOME'),
            'XDG_CACHE_HOME': os.environ.get('XDG_CACHE_HOME'),
            'PYTHONPATH': os.environ.get('PYTHONPATH')
        },
        'directories': {
            'templates_exists': os.path.exists(templates_dir),
            'templates_files': template_files,
            'temp_audio_exists': os.path.exists(temp_dir),
            'temp_files_count': len(temp_files),
            'temp_files_sample': temp_files[:5] if temp_files else []
        },
        'models_loaded': list(tts_models.keys()),
        'model_details': {
            k: {
                'correct_speaker': v['correct_speaker'],
                'sample_rate': v['sample_rate'],
                'loaded_at': v['loaded_at']
            } for k, v in tts_models.items()
        },
        'redis_connected': redis_conn.ping() if redis_conn else False,
        'numpy_version': np.__version__,
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

@app.route('/api/cleanup', methods=['POST'])
def cleanup_endpoint():
    """Очистка временных файлов"""
    try:
        temp_dir = '/app/temp_audio'
        if not os.path.exists(temp_dir):
            return jsonify({'message': 'Temp directory does not exist', 'deleted': 0})
        
        deleted_count = 0
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted_count += 1
            except:
                pass
        
        return jsonify({
            'message': f'Deleted {deleted_count} temporary files',
            'deleted': deleted_count,
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
                        # Удаляем только старые файлы (старше 1 часа)
                        if time.time() - os.path.getctime(file_path) > 3600:
                            os.remove(file_path)
                            count += 1
                    except:
                        pass
            if count > 0:
                print(f"🗑️ Удалено {count} временных файлов старше 1 часа")
        except Exception as e:
            print(f"⚠️ Ошибка очистки временных файлов: {e}")

def periodic_cleanup():
    """Периодическая очистка временных файлов"""
    while True:
        time.sleep(3600)  # Каждый час
        cleanup_temp_files()

# Регистрируем очистку при завершении
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
    print(f"🔗 Redis: {os.getenv('REDIS_HOST', 'tts-redis')}:{os.getenv('REDIS_PORT', 6379)}")
    print("=" * 70)
    
    # Запускаем периодическую очистку в фоновом потоке
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
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
    print("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )