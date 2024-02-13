# Jhana AI

Jhana AI is a voice assistant meditation coach designed to guide users through meditation sessions using advanced AI techniques. It listens to user queries, provides meditative guidance, and supports users in their meditation journey.

## Features

- **Voice Recognition**: Records audio input from the user.
- **Speech Transcription**: Utilizes Whisper to transcribe the recorded audio.
- **AI-Powered Responses**: Engages with users through Mixtral, powered by Ollama, to provide insightful meditation guidance.
- **Text-to-Speech**: Converts AI responses into audible speech using XTTS-v2.
- **Audio Playback**: Plays back the generated guidance for a seamless meditative experience.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- `ffmpeg`
- `portaudio19-dev`

In Ubuntu, you can install these dependencies using `apt`:

```bash
sudo apt install ffmpeg portaudio19-dev
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/carecodeconnect/jhana_ai
cd jhana_ai
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install [Ollama](https://ollama.com/):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Setting Up

Download the necessary models:

- **Whisper**: Follow instructions from OpenAI's [GitHub](https://github.com/openai/whisper) to download `whisper-small`.
- **TTS**: XTTS-v2 can be installed as part of the Python dependencies listed in `requirements.txt`. Test TTS in the terminal using the following command:

```bash
python -m tts --text "Hello, world!" --output-file "hello_world.wav" --model_name "tts_models/multilingual/multi-dataset/xtts_v2"
```
- **Ollama**: After installing Ollama, run the following to set up Mixtral:

```bash
ollama run mixtral:8x7b-instruct-v0.1-q4_0
```

### Folder Structure

```
jhana_ai/
│
├── data/
│   ├── input/
│   │   └── audio/
|   |       ├── speech_to_transcribe/
│   │       └── voices_to_clone/audio_cf_10_seconds.wav
|   |   
│   └── output/
│       ├── audio/
│       └── text/
│
├── notebooks/
│   ├── 01_jhana_testing.ipynb
|   └── 02_jhana_pipeline.ipynb
|
├── README.md
└── requirements.txt
```

### Basic Pipeline

1. **Record Audio**: Captures audio from the microphone and saves it as a file in `data/input/audio/speech_to_transcribe/`.
2. **Transcribe Audio**: Uses Whisper to convert the saved audio file into text.
3. **AI Interaction**: Feeds the transcribed text to Mixtral via Ollama and receives meditation guidance.
4. **Generate Response**: The response from Mixtral is printed to the console and saved in `data/output/text/`.
5. **Text-to-Speech**: Converts the Mixtral response into speech using XTTS-v2.
6. **Save and Play Speech**: The generated speech is saved in `data/output/audio/` and then played back to the user.

### Getting Started

To run the basic pipeline, run the `02_pipeline.py` Jupyter notebook in `notebooks`.

Please note the `src/main.py` file runs too slowly at the moment.
