import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa.display
import scipy.io.wavfile as wavfile

def main():
    y, sr = librosa.load('corpus/localTrainBigDataset_noAmp2/lc_train5.wav', sr=16000)
    # mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    # Find max
    n_mel = []
    for i in range(len(log_S[0])):
        col = log_S[:, i]
        max_ind = np.argmax(col)
        n_mel = np.append(n_mel, max_ind)

    # Get Labels
    labels = np.empty(n_mel.size)
    for i, item in enumerate(n_mel):
        if item <= 28:
            labels[i] = 0
        elif item > 28 and item <= 45:
            labels[i] = 1
        else:
            labels[i] = 2

    # Upsampling: Nearest Neighbour Interpolation?
    padding = int(len(y)/len(labels)/2)
    upLabels = []
    for i, item in enumerate(labels):
        padLabel = np.pad([int(labels[i])], (padding, padding-1), 'constant', constant_values=item)
        upLabels = np.append(upLabels, padLabel)
    # Fix lenghts
    if len(upLabels) > len(y):
        upLabels = upLabels[:len(y)]
    elif len(upLabels) < len(y):
        diff = len(y)-len(upLabels)
        upLabels = np.pad(upLabels, (0, diff), 'constant', constant_values=upLabels[len(upLabels)-1])

    '''
    # plot
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

    plt.show()
    '''

if __name__ == '__main__':
    main()
