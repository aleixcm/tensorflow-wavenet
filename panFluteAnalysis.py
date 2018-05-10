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

# Categories
a = [0]
b = [1]
c = [2]

def genSignals():
    y1 = x1[:8000]
    y2 = x2[:8000]
    y3 = x3[:8000]
    y4 = x4[:8000]
    y5 = x5[:8000]
    y6 = x6[:8000]

    y = scipy.hstack((y1, y2, y3, y4, y5, y6))
    y = y / y[np.argmax(y)]
    noise = 0.01*np.random.normal(0, 1, len(y))
    y = np.asarray(y) + noise
    scipy.io.wavfile.write(os.path.join('corpus', 'Analysis', 'lc_scale_flute.wav'), fs, y)

def main():
    genSignals()

if __name__ == '__main__':
    main()

