import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


file = "voice//1234"
suffix = ".wav"


def filter_audio(filename):
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    b = signal.firwin(101, [500, 3500], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    wavfile.write(filename + "_filtered" + suffix, sr, x.astype(np.int16))  # Save filtered as wav


def segment(filename):
    if "filtered" not in filename:
        if os.path.exists(filename+"_filtered"+suffix):
            filename = filename+"_filtered"
        else:
            filter_audio(filename)
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    w_size = 50
    w_offset = 12
    Zcr = librosa.feature.zero_crossing_rate(x.astype(float), 50, 12)  # Calculate zero-crossing rate
    E = []
    i = 0
    while True:
        if i*w_offset > len(x) or i*w_offset+w_size > len(x):
            E = np.array(E)
            E = 10*np.log(E)
            E = np.subtract(E, E.max())
            break
        temp = x[i*w_offset:i*w_offset+w_size].astype(np.int64)
        E.append(np.sum(np.square(temp)))
        i += 1


if __name__ == "__main__":
    segment(file)
