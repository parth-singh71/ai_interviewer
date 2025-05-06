from typing import Literal
import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
from scipy.io.wavfile import write
import threading
import queue
import tempfile
import time


# Speech-to-Text class using Whisper + WebRTC VAD
class STT:
    def __init__(
        self,
        whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "base",
        vad_mode: Literal[1, 2, 3] = 3,
    ):
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.DEFAULT_CHUNK_DURATION = 1  # seconds (used when VAD detects silence)
        self.MAX_SILENCE_SECONDS = 5  # Maximum allowed silence before stopping

        # Load Whisper model for transcription, eg: "tiny", "base", "small", "medium", "large"
        self.model = whisper.load_model(whisper_model)

        # Setup WebRTC VAD (Voice Activity Detection)
        # 0 = very sensitive
        # 1 = sensitive
        # 2 = balanced for normal speech
        # 3 = aggressive
        self.vad = webrtcvad.Vad(vad_mode)
        # Initializing session variables
        self.initialize()

    def initialize(self):
        # Reset internal buffers and states
        self.audio_queue = queue.Queue()
        self.silence_counter = 0
        self.stop_flag = threading.Event()
        self.transcript = []

    # Function to check if the current audio chunk contains speech
    def is_speech(self, audio_bytes):
        frame_duration = 30  # ms
        frame_size = int(self.SAMPLE_RATE * frame_duration / 1000) * 2  # 2 bytes/sample
        for i in range(0, len(audio_bytes) - frame_size, frame_size):
            if self.vad.is_speech(audio_bytes[i : i + frame_size], self.SAMPLE_RATE):
                return True
        return False

    # Function to transcribe audio in parallel
    def transcriber(self):
        while not self.stop_flag.is_set() or not self.audio_queue.empty():
            try:
                audio = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                write(tmpfile.name, self.SAMPLE_RATE, audio)
                result = self.model.transcribe(tmpfile.name)
                print("Transcript:", result["text"].strip())
                self.transcript.append(result["text"].strip())

    def run(self):
        print("Listening...")

        # Starting transcriber thread
        transcriber_thread = threading.Thread(target=self.transcriber, daemon=True)
        transcriber_thread.start()

        # This stores all the audio of the current sentence
        current_audio_buffer = []
        chunk_duration = self.DEFAULT_CHUNK_DURATION

        try:
            while not self.stop_flag.is_set():
                print(self.silence_counter)
                chunk_duration = self.DEFAULT_CHUNK_DURATION
                # Recording a short chunk of audio
                audio = sd.rec(
                    int(self.SAMPLE_RATE * chunk_duration),
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    dtype="int16",
                )
                sd.wait()

                audio_bytes = audio.tobytes()

                if self.is_speech(audio_bytes):
                    # Resetting silence timer and adding audio chunk to buffer
                    self.silence_counter = 0
                    print("Speech detected - adding to buffer")
                    current_audio_buffer.append(audio.copy())
                else:
                    # Silence detected
                    self.silence_counter += chunk_duration
                    print(f"Silence: {self.silence_counter} sec")

                    # If a pause is detected, transcribing the accumulated audio buffer
                    if self.silence_counter >= 1:
                        if len(current_audio_buffer) > 0:
                            print("Pause detected → transcribing sentence...")
                            # Combine the buffered audio and send to Whisper
                            full_sentence_audio = np.concatenate(
                                current_audio_buffer, axis=0
                            )
                            self.audio_queue.put(full_sentence_audio)
                            # Reset buffer for the next sentence
                            current_audio_buffer = []

                # Stopping the loop if silence exceeds the max threshold
                if self.silence_counter >= self.MAX_SILENCE_SECONDS:
                    if len(current_audio_buffer) > 0:
                        print("Pause detected -> transcribing sentence...")
                        # Combining the buffered audio and sending to Whisper
                        full_sentence_audio = np.concatenate(
                            current_audio_buffer, axis=0
                        )
                        self.audio_queue.put(full_sentence_audio)
                        # Resetting buffer for the next sentence
                        current_audio_buffer = []
                    print(
                        f"Silence for {self.MAX_SILENCE_SECONDS} seconds. Stopping..."
                    )
                    self.stop_flag.set()
                    break

        except KeyboardInterrupt:
            # Stopping recording on user interrupt
            self.stop_flag.set()
            print("\nInterrupted by user.")

        # Waiting for background thread to finish
        transcriber_thread.join()

        transcript = self.transcript
        self.initialize()
        return transcript


if __name__ == "__main__":
    stt = STT()
    print(stt.run())
