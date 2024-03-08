import ipywidgets as widgets
from IPython.display import display
import librosa
import numpy as np
import plotly.graph_objects as go
import IPython.display as ipd
import numpy as np

def visualize_audio(y, sr, play = False, L = None) -> tuple[np.ndarray, int]:
    """
    Visualize an audio signal, calculate the mean magnitude for every group of 20 samples,
    and visualize the mean magnitude over time.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sampling rate of the audio signal.
        L (np.ndarray): A 2D array where each row is a pair of values representing the start and end of a period.

    """

    # Calculate the mean magnitude for every group of 20 samples
    means = [np.mean(np.abs(y[i:i+20])) for i in range(0, len(y), 20)]
    means_db = librosa.amplitude_to_db(means)

    # Create sample count array for the means
    sample_index_means = np.arange(0, len(means)) * 20 / sr  # Convert to seconds

    # Create the plot
    fig = go.FigureWidget()
    fig.add_scatter(x=sample_index_means, y=means_db, mode='lines', name='Mean Magnitude')
    # Add a horizontal line
    fig.add_scatter(x=[sample_index_means[0], sample_index_means[-1]], y=[min(means_db), min(means_db)], mode='lines', name='Threshold')

    # Add vertical lines for each value in L
    if L is not None:
        for start, end in L:
            # Add a line at the start of the period
            fig.add_shape(
                type="line",
                x0=start,
                y0=min(means_db),
                x1=start,
                y1=max(means_db),
                line=dict(
                    color="Red",
                    width=2,
                    dash="dashdot",
                ),
            )
            # Add a line at the end of the period
            fig.add_shape(
                type="line",
                x0=end,
                y0=min(means_db),
                x1=end,
                y1=max(means_db),
                line=dict(
                    color="Red",
                    width=2,
                    dash="dashdot",
                ),
            )

    fig.update_layout(
        title='Mean Magnitude over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Mean Magnitude (dB)',
        width=1200,  # Width of the figure in pixels
        height=600   # Height of the figure in pixels
    )

    # Create a slider widget
    slider = widgets.FloatSlider(value=min(means_db), min=min(means_db), max=max(means_db), step=0.01,
                                 description='Threshold:', continuous_update=True)

    # Plus and minus buttons
    btn_plus = widgets.Button(description='+')
    btn_minus = widgets.Button(description='-')

    # Function to update the figure based on the slider
    def update_line(change):
        with fig.batch_update():
            fig.data[1].y = [change.new, change.new]

    slider.observe(update_line, names='value')

    # Button click events
    def on_plus_clicked(b):
        slider.value += 10*slider.step

    def on_minus_clicked(b):
        slider.value -= 10*slider.step

    btn_plus.on_click(on_plus_clicked)
    btn_minus.on_click(on_minus_clicked)

    # Display everything
    controls = widgets.HBox([btn_minus, slider, btn_plus])
    display(widgets.VBox([controls, fig]))

    if play == True:
        # listen to the audio file
        display(ipd.Audio(y, rate=sr))

if __name__ == '__main__':
    load_and_visualize_audio('longer_example.wav')
