import numpy as np
import time
import pyaudio
from openwakeword.model import Model

CHANNELS = 1
MODEL_PATH = "yah_zuh_juh.tflite"
CHUNK = 1280
RATE = 16000
FORMAT = pyaudio.paInt16
HOTWORD_SENSITIVITY = 0.1
audio = pyaudio.PyAudio()
owwModel = None
mic_stream = None
is_listening = False
hotword_thread = None
last_detect_time = 0
cooldown = 2.0

def hotword_listener():
    """Hotword dinler ve mikrofonu SOX'a devreder. (Hayalet tetikleme düzeltildi)"""
    global last_detect_time, mic_stream, owwModel

    # Modeli döngü dışında bir kere yükle
    owwModel = Model(wakeword_models=[MODEL_PATH], inference_framework="tflite")
    
    while is_listening:
        # --- Python Mikrofonu Alır ---
        try:
            mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                   input=True, frames_per_buffer=CHUNK)
            
            # YENİ EKLENEN KRİTİK SATIR: Modelin hafızasını temizle
            owwModel.reset()
            
            print("\n🎧 Hotword dinleniyor...")
            #status_label.config(text="🎧 Hotword dinleniyor...", fg="green")
            
        except Exception as e:
            print(f"❌ Mikrofon açılamadı, 5 saniye sonra tekrar denenecek. Hata: {e}")
            time.sleep(5)
            continue

        hotword_detected = False
        while is_listening and not hotword_detected:
            try:
                audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                owwModel.predict(audio_data)
                
                for mdl in owwModel.prediction_buffer.keys():
                    score = list(owwModel.prediction_buffer[mdl])[-1]
                    if score > HOTWORD_SENSITIVITY and (time.time() - last_detect_time > cooldown):
                        print(f"🎯 Hotword algılandı! Skor: {score:.3f}")
                        last_detect_time = time.time()
                        hotword_detected = True
                        break
            except IOError:
                pass
            except Exception as e:
                print(f"❌ Hotword dinleme hatası: {e}")
                break
        
        # --- Python Mikrofonu Bırakır ---
        mic_stream.stop_stream()
        mic_stream.close()
        

        if hotword_detected and is_listening:
            print("Hotword detected")

    
    print("Hotword dinleme döngüsü sonlandı.")
    
    
if __name__ == "__main__":
    is_listening = True
    try:
        hotword_listener()
    except KeyboardInterrupt:
        is_listening = False
        print("Dinleme durduruldu.")

    
