import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from segmentation import preprocess
from dtw import dtw, accelerated_dtw

prefix = "voice//"
filename = "1234_filtered_seg_4"
suffix = ".wav"
SAVE = 0

if __name__ == "__main__":
    sr, x = wavfile.read(prefix + filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    #b = signal.firwin(101, [500, 3500], fs=sr, pass_zero=False)  # Create a FIR band pass filter
    #x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (sr / 1000))  # Window size in samples
    R = int(MS * (sr / 1000))  # Window offset in samples
    feat = []
    i = 0
    while True:
        feat.append(librosa.feature.mfcc(x[i*R:i*R+R].astype(float))[1:])
        i += 1
        if len(x) - (i*R+R) == 0:
            break
        elif len(x) - (i*R+R) < 0:
            feat.append(librosa.feature.mfcc(x[i*R:].astype(float))[1:])
            break
    feat = np.array(feat)
    feat = np.reshape(feat, (feat.shape[0], feat.shape[1]))
    if SAVE:
        np.save(prefix+"mfcc//"+filename, feat)
    else:
        mat = []
        for i in range(1, 8):
            y = np.load("voice//mfcc//"+"raw1234567_seg_{}".format(i)+".npy")
            mat.append(accelerated_dtw(feat, y, dist='euclidean')[0])
        print(mat.index(min(mat))+1)

