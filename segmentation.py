import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

file = "voice//M1"
suffix = ".wav"


def preprocess(sr, x):
    if x.shape[0] == 2:
        x = librosa.to_mono(x.astype(float))
    if sr != 8000:
        x = librosa.resample(x.astype(float), sr, 8000)
        sr = 8000
    return sr, x


def filter_audio(filename):
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    b = signal.firwin(101, [200, 3800], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    wavfile.write(filename + "-f" + suffix, sr, x.astype(np.int16))  # Save filtered as wav
    return x


def segment(filename):
    suffix = ".wav"
    if suffix in filename:
        filename, suffix = filename.split('.')
        suffix = '.' + suffix
    filter_audio(filename)
    sr, x = wavfile.read(filename + "-f" + suffix)  # Load the audio clip

    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (sr/1000))  # Window size in samples
    R = int(MS * (sr/1000))  # Window offset in samples

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

    IMX = -30
    ITU = IMX - 50  # High threshold for E
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
                # print(ref_voice_pos)
                break

    sr, x = wavfile.read(filename + suffix)
    prefix, filename = filename.split('//')
    prefix += '//segmented//'
    # Save the segmented voice clips using the unfiltered file as a base
    for i, pos in enumerate(ref_voice_pos):
        start_idx = pos[0] * R
        end_idx = pos[1] * R
        wavfile.write(prefix + filename + "-{}".format(i) + suffix, sr, x[start_idx:end_idx].astype(np.int16))

    return ref_voice_pos


if __name__ == "__main__":
    segment(file)
