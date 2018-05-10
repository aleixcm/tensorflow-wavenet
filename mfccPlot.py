import matplotlib.pyplot as plt
import librosa.display
from sklearn import preprocessing


#np.set_printoptions(threshold=np.nan)
y, sr = librosa.load('corpus/localTlc_train4.wav', sr=16000)

def calculateMFCC(y,sr):
    mfccs = librosa.feature.mfcc(y, sr=sr)
    mfccs = preprocessing.scale(mfccs, axis=1)
    return(mfccs)


def main():
    mfccs=calculateMFCC(y,sr)

    plt.figure()
    for i in range(0,12):

        plt.plot(upLabels12[i][:])

    plt.show()
    #plots
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('MFCCs')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

if __name__ == '__main__':
    main()