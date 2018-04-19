import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import os

fs = 16000

# Categories
a = [0]
b = [1]
c = [2]

# Samples per Category
aSamples = 8000
bSamples = 8000
cSamples = 8000

aSamplesV = np.arange(aSamples) #length in samples
bSamplesV = np.arange(bSamples) #length in samples
cSamplesV = np.arange(cSamples) #length in samples

# Frequencies
fa = 440
fb = 880
fc = 1320


def genFile(a, b, c):
    sequence = a * aSamples + b * bSamples + c * cSamples
    file00 = open(os.path.join('corpus', 'localTrain', 'lc_train0.txt'), 'w')
    for item in sequence:
        file00.write('%i,\n' % item)

    file00.close()

def genSignals(fa, fb, fc, aSamplesV, bSamplesV, cSamplesV):
    y1 = np.sin(2 * np.pi * fa * aSamplesV / fs)
    y2 = np.sin(2 * np.pi * fb * bSamplesV / fs)
    y3 = np.sin(2 * np.pi * fc * cSamplesV / fs)
    y=np.concatenate((y1, y2, y3))
    scipy.io.wavfile.write(os.path.join('corpus', 'localTrain', 'lc_train0.wav'), fs, y)
    #plt.figure(1)
    #plt.plot(y)
    #plt.show()

def main():
    genFile(a, b, c)
    genSignals(fa, fb, fc, aSamplesV, bSamplesV, cSamplesV)

if __name__ == '__main__':
    main()