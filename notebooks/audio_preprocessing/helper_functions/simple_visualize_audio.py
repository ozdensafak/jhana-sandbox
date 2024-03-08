import numpy as np
import librosa

# This is a helper file that contains functions to visualize audio data.

# The first function, visualize_audio, takes in an audio file and plots the amplitude of the audio signal over time. This is useful for understanding the structure of the audio data and identifying any patterns or anomalies.
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import librosa
import plotly.graph_objects as go

def visualize_audio_continuous(audio_file: str) -> tuple:
    """
    Loads an audio file, visualizes the amplitude of the audio signal over time, and returns the audio data and sample rate.

    Parameters:
    audio_file (str): The path to the audio file.

    Returns:
    tuple: A tuple containing the audio data and sample rate.
    """
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Calculate amplitude envelope
    amplitude_envelope = librosa.amplitude_to_db(abs(y))

    # Create sample count array
    sample_index = librosa.frames_to_time(range(len(y)), sr=sr)

    # Create plot
    fig = go.Figure(data=go.Scatter(x=sample_index, y=amplitude_envelope))
    fig.update_layout(title='Loudness over time', xaxis_title='Sample index', yaxis_title='Amplitude (dB)')
    fig.show()

    return y, sr

# The second function visualize the amplitude mean with window size of 10
def visualize_audio(audio_file: str) -> None:
    """
    Visualizes the amplitude of the audio signal over time.

    Parameters:
    audio_file (str): The path to the audio file.
    """
    # Load audio file
    y, _ = librosa.load(audio_file)

    # Calculate amplitude envelope
    amplitude_envelope = librosa.amplitude_to_db(abs(y))

    # Calculate amplitude mean with window size of 10
    amplitude_mean = np.mean(amplitude_envelope.reshape(-1, 10), axis=1)

    # Create time array
    time = np.arange(0, len(amplitude_mean))

    # Create plot
    plt.plot(time, amplitude_mean)
    plt.title('Loudness over time')
    plt.xlabel('Time')
    plt.ylabel('Amplitude (dB)')
    plt.show()


if __name__ == "__main__":
    audio_file = "../longer_example.wav" 
    visualize_audio(audio_file)



