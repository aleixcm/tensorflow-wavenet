import matplotlib.pyplot as plt
import librosa.display
from sklearn import preprocessing


#np.set_printoptions(threshold=np.nan)
y1, sr = librosa.load('corpus/Analysis/p225_001.wav', sr=16000)
y2, sr = librosa.load('corpus/Analysis/p225_001.wav', sr=16000)

def calculateMFCC(y,sr):
    mfccs = librosa.feature.mfcc(y, n_mfcc=13, sr=sr)
    mfccs = preprocessing.scale(mfccs, axis=1)
    return(mfccs)

def calculateMean(mfccs1, mfccs2):
    mfccs = (mfccs1+mfccs2)/2
    return(mfccs)

def main():
    mfccs1 =calculateMFCC(y1,sr)
    mfccs2 = calculateMFCC(y2, sr)
    mfccs = calculateMean(mfccs1, mfccs2)

    #plots
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs1, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('MFCCs')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs2, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('MFCCs')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('MFCCs')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()