import librosa
import numpy as np
from scipy.io import wavfile
from segmentation import preprocess, segment
import glob
from dtw import accelerated_dtw

prefix = "voice//"
filename = "M1"
suffix = ".wav"

SAVE = 0  # If set to 1 the program updates the "database" and doesn't do the recognition
N_MFCC = 18  # Number of MFCCs calculated we found out that 18 was a sweet spot

if __name__ == "__main__":
    sr, x = wavfile.read(prefix + filename + suffix)  # Load the audio clip with native sampling rate
    sr, x = preprocess(sr, x)
    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (sr / 1000))  # Window size in samples
    R = int(MS * (sr / 1000))  # Window offset in samples
    x_segs = segment(prefix + filename + suffix)

    for j, seg in enumerate(x_segs):
        feat = []
        y = x[seg[0]:seg[1]]
        i = 0
        hamm_wind = np.hamming(L)  # Make a hamming window
        while True:
            frame = np.multiply(y[i*R:i*R+L], hamm_wind)
            feat.append(librosa.feature.mfcc(frame.astype(float), n_mfcc=N_MFCC)[1:])

            i += 1
            if len(y) - (i*R+L) == 0:
                break
            elif len(y) - (i*R+L) < 0:
                frame = np.multiply(y[i*R:], np.hamming(len(y)-i*R))
                feat.append(librosa.feature.mfcc(y[i*R:].astype(float), n_mfcc=N_MFCC)[1:])
                break
        feat = np.array(feat)

        feat = np.reshape(feat, (feat.shape[0], feat.shape[1]))
        if SAVE:
            np.save(prefix+"mfcc//"+filename+"-{}".format(j), feat)
        else:
            mat = []
            mat_file = []
            for it in glob.glob(prefix + 'mfcc//*'):
                if filename in it:
                    continue
                z = np.load(it)
                mat.append(accelerated_dtw(feat, z, dist='euclidean')[0])
                mat_file.append(it[12:-4])
            #
            # Prints the filename which is the most similar for each sound segment
            # By printing M1-0 for example the program found that M1-0 is the most
            # similar and by extension means that the number which was pronounced
            # was "zero", M1-1 would be "one", M1-2 would be "two" and so forth.
            #
            print(mat_file[mat.index(min(mat))][3:])

