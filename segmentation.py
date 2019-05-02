import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


filename = "voice//1234"
suffix = ".wav"


def segment(filename):
    sr, x = wavfile.read(filename+suffix)  # Load the audio clip with native sampling rate
    b = signal.firwin(101, [500, 3500], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    wavfile.write(filename+"_filtered"+suffix, sr, x.astype(np.int16))  # Save filtered as wav


if __name__ == "__main__":
    segment(filename)
