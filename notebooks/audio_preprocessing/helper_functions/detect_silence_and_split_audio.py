from os import name
import numpy as np
import librosa


def detect_periods(y: np.ndarray, sr: int, threshold: float) -> tuple:
    """
    Detect the periods of non-silence in the audio signal based on a given threshold.

    Parameters:
    y (np.ndarray): The audio signal.
    sr (int): The sample rate of the audio signal.
    threshold (float): The threshold value for detecting non-silence.

    Returns:
    tuple: A tuple containing three arrays. 
        1st array: Non-silence .
        2nd array: Silent periods.
        3rd array: Periods as dictionaries with 'key = non-silence/silence' and 'value' fields.

    """
    unfiltered_non_silent_periods = librosa.effects.split(y, top_db=threshold)/sr
    silence = []
    for i in range(unfiltered_non_silent_periods.shape[0]-1):
        if unfiltered_non_silent_periods[i+1, 0] -unfiltered_non_silent_periods[i, 1] > 2:
            silence.append([unfiltered_non_silent_periods[i, 1]+0.5, unfiltered_non_silent_periods[i+1, 0]-0.5])

    silence = np.array(silence)  # Convert to numpy array

    # return complement of silence
    audio_length = len(y)/sr
    if silence.size > 0:  # Check if silence is not empty
        non_silence = np.array([[0, silence[0, 0]]])
        for i in range(silence.shape[0]-1):
            non_silence = np.append(non_silence, [[silence[i, 1], silence[i+1, 0]]], axis=0)
        non_silence = np.append(non_silence, [[silence[-1, 1], audio_length]], axis=0)
    else:
        non_silence = np.array([[0, audio_length]])

    # Merge non_silence and silence arrays and sort by start time
    periods = np.concatenate((non_silence, silence))
    periods = periods[periods[:, 0].argsort()]

    # Create set of non-silent periods
    non_silence_set = set(map(tuple, non_silence))

    # Create list of dictionaries
    # periods_dicts = [{'key': 'non-silence' if tuple(period) in non_silence_set else 'silence', 'value': period} for period in periods]
    # Create list of dictionaries
    periods_dicts = [{('non-silence' if tuple(period) in non_silence_set else 'silence'): period} for period in periods]

    return non_silence, silence, periods_dicts

def split_audio(y: np.ndarray, sr: int, periods: np.ndarray) -> list:
    """
    Split the audio signal into segments based on the specified periods.

    Parameters:
    y (np.ndarray): The audio signal.
    sr (int): The sample rate of the audio signal.
    periods (np.ndarray): The periods indicating the start and end points of each segment.

    Returns:
    list: A list of audio segments.

    """
    audio_segments = []
    for start, end in periods:
        audio_segments.append(y[int(start*sr):int(end*sr)])
    return audio_segments


if __name__ == "__main__":
    pass