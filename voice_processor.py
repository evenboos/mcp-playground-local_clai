import threading
import pyaudio
import numpy as np
from funasr import AutoModel
from config import VOICE_WAKEUP_WORD

class VoiceProcessor:
    def __init__(self, callback):
        self.callback = callback
        self._running = False
        self._thread = None
        self.activated = False
        self.previous_text = ""
        self.cache = {}
        self.chunk_size_list = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        self.model = AutoModel(model="paraformer-zh-streaming",
                               device="cuda",
                               disable_pbar=True)
        
        self.p = None
        self.CHUNK = 9600 # 960 samples for 600ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.stream = None

    def start(self):
        if not self._running:
            self.p = pyaudio.PyAudio()
            self._running = True
            self._thread = threading.Thread(target=self._processing_loop)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        if self.p:
            self.p.terminate()
            self.p = None
            print("PyAudio terminated.")

    def _processing_loop(self):
        try:
            self.stream = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK)
            
            device_info = self.p.get_default_input_device_info()
            device_name = device_info['name']
            print(f"🎤 Listening on microphone: {device_name}")

            while self._running:
                data = self.stream.read(self.CHUNK)
                data = np.frombuffer(data, dtype=np.int16)
                rec_result = self.model.generate(
                    input=data,
                    cache=self.cache,
                    is_final=False,
                    chunk_size=self.chunk_size_list,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )
                
                text = rec_result[0].get('text') if rec_result and rec_result[0] else ""
                if text:
                    print(f"Recognized: '{text}', Activated: {self.activated}")

                if not self.activated:
                    # --- New robust wake word detection logic ---
                    combined_text = self.previous_text + text
                    if VOICE_WAKEUP_WORD in combined_text:
                        self.activated = True
                        print("Wake word detected!")
                        self.previous_text = "" # Clear buffer after activation
                    else:
                        self.previous_text = text # Update buffer for next chunk
                    # --- End of new logic ---
                else:
                    if text:
                        self.callback.put({"text": text})
                    # VAD (Voice Activity Detection) logic
                    # If VAD result is empty, it implies silence.
                    is_final = rec_result[0].get('is_final', False)
                    if not text and self.activated:
                        print("Silence detected, submitting.")
                        self.callback.put({"action": "submit"})
                        self.activated = False
                        self.previous_text = "" # Clear buffer on deactivation
                        self.cache = {}
        except OSError as e:
            print(f"无法访问麦克风设备: {e}")
            print("请检查您的麦克风是否连接正常，或在系统设置中授予了访问权限。")
            return

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("Microphone stream closed.")
