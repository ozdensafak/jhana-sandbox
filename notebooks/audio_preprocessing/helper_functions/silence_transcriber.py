import soundfile as sf
import whisper
import librosa
import tempfile
import numpy as np


def transcribe_silence(y: np.ndarray, sr, periods_dictionaries: list, file_name: str, whisper_name="small", language="en"):
    """
    1. Opens a txt.file to write transcriptions
    2. Iterates through all periods
    3. If the period is silence, it writes "silence: ceiling of the length of the period in seconds"
    4. If the period is non-silence, it transcribes the audio and writes the transcription
    5. Each trascription is written in a new line
    6. Save the file to file;_name and close the file

    params:
    y: numpy array, the original audio file that will be transcribed
    sr: int, the sample rate of the audio file
    periods_dictionaries: list of dictionaries, each period is a dictonary one one key value pair
            period key: either silence or non-silence
            period value: numpy array indicating the start and end time (in seconds)
    file_name: str, the name of the file where the transcriptions will be saved
    whisper_name: str, the name of the whisper model to be used for the transcription
    language: str, the language of the audio file
    """
    model = whisper.load_model(whisper_name)

    with open(f"{file_name}.txt", "w") as file:
        for period in periods_dictionaries:
            period_type = next(iter(period.keys()))
            start, end = next(iter(period.values()))
            if period_type == "silence":
                file.write(f"silence: {np.ceil(end - start)}\n")
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(tmpfile, y[int(start*sr):int(end*sr)], sr)
                    result = model.transcribe(tmpfile.name, language=language)
                    file.write(result["text"] + "\n")       


if __name__ == "__main__":
    pass