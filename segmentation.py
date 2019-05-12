import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os


file = "voice//A09"
suffix = ".wav"


def preprocess(sr, x):
    x = x.T
    if x.shape[0] == 2:
        x = librosa.to_mono(x.astype(float))
    if sr != 8000:
        x = librosa.resample(x.astype(float), sr, 8000)
    return 8000, x


def filter_audio(filename):
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    b = signal.firwin(101, [500, 3500], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    wavfile.write(filename + "_filtered" + suffix, sr, x.astype(np.int16))  # Save filtered as wav


def segment(filename):
    suffix = ".wav"
    if ".wav" in filename:
        filename, suffix = filename.split('.')
        suffix = '.' + suffix
    if "filtered" not in filename:
        if os.path.exists(filename+"_filtered"+suffix):
            filename = filename+"_filtered"
        else:
            filter_audio(filename)
            filename = filename + "_filtered"
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)

    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (sr/1000))  # Window size in samples
    R = int(MS * (sr/1000))  # Window offset in samples

    Zc = librosa.feature.zero_crossing_rate(x.astype(float), R, R)  # Calculate zero-crossing rate
    Zc = np.reshape(np.multiply(Zc, R), (Zc.shape[1],))
    E = []
    i = 0
    while True:
        temp = x[i*R:i*R+L].astype(np.int64)
        E.append(np.sum(np.square(temp)))
        i += 1
        if i*R > len(x) or i*R+L > len(x):
            E = np.array(E)
            E = 10*np.log(E)
            E = np.subtract(E, E.max())
            break
    eavg = np.mean(E[:10])
    esig = np.std(E[:10])
    zcavg = np.mean(Zc[:10])
    zcsig = np.std(Zc[:10])

    IF = 35  # Fixed threshold for Zc
    IZCT = max(IF, zcavg + 2*zcsig)  # Variable threshold for Zc
    IMX = -30
    ITU = IMX - 20  # High threshold for E
    ITL = max(eavg+3*esig, ITU-10)  # Low threshold for E

    end_idx = 0
    voice_pos = []

    # Find the multi-frames where the logarithmic energy is above ITU and save them in a list
    for i, val in enumerate(E):
        if i < end_idx:
            continue
        if val >= ITL:
            start_idx = i
            for j, val2 in enumerate(E[i:]):
                if val2 <= ITL:
                    end_idx = j + start_idx
                    voice_pos.append([start_idx, end_idx])
                    break

    # Look through voice_pos and merge elements which are no longer than 100 frames long
    next = True
    ref_voice_pos = []
    for i, tup in enumerate(voice_pos):
        if next:
            start_idx = tup[0]
            next = False
        if tup[1] - start_idx < 30:
            continue
        else:
            try:
                if voice_pos[i+1][1] - start_idx > 100:
                    end_idx = voice_pos[i][1]
                    ref_voice_pos.append([start_idx, end_idx])
                    next = True
                    continue
            except IndexError:
                end_idx = voice_pos[i][1]
                ref_voice_pos.append([start_idx, end_idx])
                print(ref_voice_pos)
                break

    # Save the segmented voice clips using the unfiltered file as a base
    for i, pos in enumerate(ref_voice_pos):
        sr, x = wavfile.read(file + suffix)
        start_idx = pos[0]*R
        end_idx = pos[1]*R
        wavfile.write(file + "_seg_{}".format(i+1) + suffix, sr, x[start_idx:end_idx].astype(np.int16))


if __name__ == "__main__":
    segment(file)
