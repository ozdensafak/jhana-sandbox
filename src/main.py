import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import ollama
import torch
from TTS.api import TTS
import subprocess  # Import subprocess to use aplay

# Function to record audio
def record_audio(duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait()  # Wait until recording is finished
    recording = np.int16(recording / np.max(np.abs(recording)) * 32767)  # Convert to int16
    return recording, fs

# Record audio
output_directory = "../data/input/audio/speech_to_transcribe"
os.makedirs(output_directory, exist_ok=True)
audio, fs = record_audio(duration=5)
audio_file_path = os.path.join(output_directory, "my_voice_recording.wav")
write(audio_file_path, fs, audio)
print(f"Recording saved to {audio_file_path}")

# Convert speech to text
model = whisper.load_model("small")
result = model.transcribe(audio_file_path, language="en")
transcribed_text = result["text"]
print("Transcribed text:", transcribed_text)

# Chat with Ollama
ollama_response = ollama.chat(model='mixtral:8x7b-instruct-v0.1-q4_0', messages=[{'role': 'user', 'content': transcribed_text}])
ollama_text = ollama_response['message']['content']
print("Ollama response:", ollama_text)

# Save Ollama's response as text
output_text_directory = "../data/output/text/"
os.makedirs(output_text_directory, exist_ok=True)
text_file_path = os.path.join(output_text_directory, "ollama_response.txt")
with open(text_file_path, "w") as text_file:
    text_file.write(ollama_text)
print(f"Ollama's response saved to {text_file_path}")

# Convert Ollama's response to speech
device = "cpu"  # Force the operation to use the CPU to avoid CUDA memory issues
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)  # Adjust model as needed
output_audio_directory = "../data/output/audio/"
os.makedirs(output_audio_directory, exist_ok=True)
output_file_path = os.path.join(output_audio_directory, "ollama_response.wav")
tts.tts_to_file(text=ollama_text, file_path=output_file_path, language="en", speaker_wav="../data/input/audio/voices_to_clone/audio_cf_10_seconds.wav")
print(f"Text-to-speech audio saved to {output_file_path}")

# Play the generated speech using aplay
if os.path.exists(output_file_path):
    subprocess.run(["aplay", output_file_path])
else:
    print("Audio file not found.")

 