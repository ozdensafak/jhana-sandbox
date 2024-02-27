import os
import asyncio
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import ollama
print(torch.version.cuda)

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)

# Function to record audio asynchronously
async def record_audio_async(duration=5, fs=44100):
    print("Please start speaking now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    await asyncio.sleep(duration)
    print("Recording complete.")
    recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
    return recording, fs

async def transcribe_audio_async(audio_file_path):
    result = whisper_model.transcribe(audio_file_path, language="en")
    if device == "cuda":
        torch.cuda.empty_cache()
    return result["text"]

async def chat_with_ollama_async(transcribed_text):
    ollama_response = ollama.chat(model='mixtral:8x7b-instruct-v0.1-q4_0', messages=[{'role': 'user', 'content': transcribed_text}])
    return ollama_response['message']['content']

async def stream_audio_response(text):
    print("Loading XTTS model...")
    config = XttsConfig()
    config.load_json("/home/solaris/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/home/solaris/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/", use_deepspeed=True)
    model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])

    print("Streaming inference...")
    chunks = model.inference_stream(text, "en", gpt_cond_latent, speaker_embedding)

    for i, chunk in enumerate(chunks):
        print(f"Streaming chunk {i} of audio length {chunk.shape[-1]}")
        sd.play(chunk.cpu().numpy(), 24000)
        sd.wait()

async def main():
    output_directory = "../data/input/audio/speech_to_transcribe"
    os.makedirs(output_directory, exist_ok=True)
    
    audio, fs = await record_audio_async(duration=5)
    audio_file_path = os.path.join(output_directory, "my_voice_recording.wav")
    write(audio_file_path, fs, audio)
    print(f"Recording saved to {audio_file_path}")

    transcribed_text = await transcribe_audio_async(audio_file_path)
    print("Transcribed text:", transcribed_text)

    ollama_response = await chat_with_ollama_async(transcribed_text)
    print("Ollama response:", ollama_response)

    await stream_audio_response(ollama_response)

if __name__ == "__main__":
    asyncio.run(main())

