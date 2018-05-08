import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa.display
import scipy.io.wavfile as wavfile

def main():
    y, sr = librosa.load('corpus/panFluteBigDataset7freq/lc_train0.wav', sr=16000)
    # mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    # Find max
    labels = []
    labelsNorm = []
    #for i in range(len(S[0])): in case that we want to calculate the mean use S

    for i in range(len(log_S[0])):
        #col = S[:, i] in case that we want to calculate the mean use S
        col = log_S[:, i]

        # If we calculate the mean per section, in yes dataset is always 0. We couldn't condition in that way
        #ampC0 = np.mean(col[:43])
        #ampC1 = np.mean(col[43:86])
        #ampC2 = np.mean(col[86:128])

        #amp = np.append(ampC0, ampC1)
        #amp = np.append(amp, ampC1)
        #label = np.argmax(amp)
        #labels = np.append(labels, label)

        '''
        # Working for three categories
        ampIndex = np.argmax(col)
        if ampIndex <= 28:
            label = 0
        elif ampIndex > 28 and ampIndex <=45:
            label = 1
        else:
            label = 2

        labels = np.append(labels, label)
        '''
        ampIndex = np.argmax(col)
        labels = np.append(labels,ampIndex)
        labelsUniq = np.unique(labels)
        labelsIndices=[]
    for index, labelsEnum in enumerate(labelsUniq):
        labelsIndices = np.append(labelsIndices, index)

    for item in labels:
        for i in range(len(labelsUniq)):
            if item==labelsUniq[i]:
                labelNorm=labelsIndices[i]
                labelsNorm = np.append(labelsNorm, labelNorm)

    labels = labelsNorm
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

    #plot

    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    #librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
