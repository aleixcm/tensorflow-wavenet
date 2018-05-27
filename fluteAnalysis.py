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

x1 = scipy.io.wavfile.read('corpus/Analysis/a440.wav')[1]
x2 = scipy.io.wavfile.read('corpus/Analysis/c531.wav')[1]
x3 = scipy.io.wavfile.read('corpus/Analysis/e667.wav')[1]
x4 = scipy.io.wavfile.read('corpus/Analysis/a880.wav')[1]
x5 = scipy.io.wavfile.read('corpus/Analysis/c1056.wav')[1]
x6 = scipy.io.wavfile.read('corpus/Analysis/e1320.wav')[1]
x7 = scipy.io.wavfile.read('corpus/Analysis/a1760.wav')[1]

def case(x):
    return {
        0: x1,
        1: x2,
        2: x3,
        3: x4,
        4: x5,
        5: x6,
        6: x7
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
    scipy.io.wavfile.write(os.path.join('corpus', 'Analysis', 'lc_gen4_flute.wav'), fs, y)

def main():
    sequence = [0, 3, 5]
    sampleSequence = [24000, 24000, 24000]
    genSignals(sequence, sampleSequence)

if __name__ == '__main__':
    main()

