import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import random


'''
Create a random dataset with three different frequencies that are always in fase.
Frequencies will be octave [440, 880, 1320].
'''

fs = 16000

x1 = scipy.io.wavfile.read('corpus/piano/a440.wav')[1]
x2 = scipy.io.wavfile.read('corpus/piano/a880.wav')[1]
x3 = scipy.io.wavfile.read('corpus/piano/e1320.wav')[1]

def case(x):
    return {
        0: x1,
        1: x2,
        2: x3,
    }[x]

def genSignals(sequence, sampleSequence):
    y=[]
    for i in range(len(sequence)):
        # convert categories to frequencies
        freq = case(sequence[i])
        #nSamples = np.arange(sampleSequence[i])
        #a = random.randint(25, 100)/100
        a = 1
        #y0 = a*np.sin(2*np.pi*freq*nSamples / fs)
        y0= freq[:sampleSequence[i]]
        y = scipy.hstack((y, y0))

    y = y / y[np.argmax(y)]
    noise = 0.01*np.random.normal(0, 1, len(y))
    y = np.asarray(y) + noise
    scipy.io.wavfile.write(os.path.join('corpus', 'Analysis', 'lc_gen3_piano.wav'), fs, y)

def main():
    sequence = [2, 1, 0]
    sampleSequence = [16000, 16000, 16000]
    genSignals(sequence, sampleSequence)

if __name__ == '__main__':
    main()

