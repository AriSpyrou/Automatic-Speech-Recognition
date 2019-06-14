import librosa
import numpy as np
from scipy.io import wavfile
from segmentation import preprocess
import glob
from dtw import dtw, accelerated_dtw

prefix = "voice//segmented//"
filename = "F1-9"
suffix = ".wav"
SAVE = 0

if __name__ == "__main__":
    sr, x = wavfile.read(prefix + filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    # b = signal.firwin(101, [500, 3500], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    # x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (sr / 1000))  # Window size in samples
    R = int(MS * (sr / 1000))  # Window offset in samples
    feat = []
    i = 0
    while True:
        frame = np.multiply(x[i*R:i*R+L], np.hamming(L))
        feat.append(librosa.feature.mfcc(frame.astype(float))[1:])
        i += 1
        if len(x) - (i*R+L) == 0:
            break
        elif len(x) - (i*R+L) < 0:
            frame = np.multiply(x[i*R:], np.hamming(len(x)-i*R))
            feat.append(librosa.feature.mfcc(x[i*R:].astype(float))[1:])
            break
    feat = np.array(feat)

    feat = np.reshape(feat, (feat.shape[0], feat.shape[1]))
    if SAVE:
        np.save(prefix+"mfcc//"+filename, feat)
    else:
        mat = []
        for it in glob.glob(prefix + 'mfcc//*'):
            if filename[:-1] in it:
                continue
            y = np.load(it)
            mat.append(accelerated_dtw(feat, y, dist='euclidean')[0])
        print(mat.index(min(mat)))

