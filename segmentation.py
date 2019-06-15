import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt

file = "voice//M2"
suffix = ".wav"

SAVE_FIL = 0  # Save filtered audio to file
SAVE_SEGS = 1  # Save segmented audio clips for debugging and evaluation


def preprocess(sr, x):
    try:
        if x.shape[1] == 2:
            x = x.T
            x = librosa.to_mono(x.astype(float))
    except IndexError:
        pass
    if sr != 8000:
        x = librosa.resample(x.astype(float), sr, 8000)
        sr = 8000
    return sr, x


def filter_audio(filename, band_pass):
    sr, x = wavfile.read(filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    b = signal.firwin(101, band_pass, fs=sr, pass_zero=False)  # Create a FIR band pass filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    if SAVE_FIL:
        wavfile.write(filename + "-f" + suffix, sr, x.astype(np.int16))  # Save filtered as wav
    return sr, x


def segment(filename):
    suffix = ".wav"
    if suffix in filename:
        filename, suffix = filename.split('.')
        suffix = '.' + suffix
    sr, x = filter_audio(filename, [200, 3800])

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
            #plt.plot(E)
            #plt.show()
            break
    os = 0
    for val in E:
        if val == -np.inf:
            os += 1
        else:
            break
    eavg = np.mean(E[os:10+os])
    esig = np.std(E[os:10+os])

    ITL = max(eavg+3*esig, -60)  # Threshold for E

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
                break

    sr, x = wavfile.read(filename + suffix)
    prefix, filename = filename.split('//')
    prefix += '//segmented//'
    ref_voice_pos = np.multiply(ref_voice_pos, R)
    # Save the segmented voice clips using the unfiltered file as a base
    if SAVE_SEGS:
        for i, pos in enumerate(ref_voice_pos):
            wavfile.write(prefix + filename + "-{}".format(i) + suffix, sr, x[pos[0]:pos[1]].astype(np.int16))
    return ref_voice_pos


if __name__ == "__main__":
    segment(file)
