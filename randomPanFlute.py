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

def createRandomSequence():
    # sequence length
    sq_length = random.randint(5, 10)
    #create sequence
    sequence = []
    sampleSequence = []
    minLen = 1818
    for i in range(0, sq_length):
        value = random.randint(0,6)
        sequence.append(value)
        #create lengths per value
        lenValue = minLen * random.randint(1,10)
        sampleSequence.append(lenValue)

    return sequence, sampleSequence


def genFile(sequence, sampleSequence, c):
    newSequence = []
    fullSequence = []
    for i in range(len(sequence)):
        newSequence = int(sampleSequence[i]) * [sequence[i]]
        fullSequence = fullSequence + newSequence

    file00 = open(os.path.join('corpus', 'panFluteBigDataset', 'lc_train%s.txt' % c), 'w')
    for item in fullSequence:
        file00.write('%i,\n' % item)

    file00.close()

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

def genSignals(sequence, sampleSequence, c):
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
    scipy.io.wavfile.write(os.path.join('corpus', 'panFluteBigDataset7freq', 'lc_train%s.wav' % c), fs, y)

def main():
    for c in range(0,100):
        sequence, sampleSequence = createRandomSequence()
        #print(sequence, sampleSequence)
        #genFile(sequence, sampleSequence, c)
        genSignals(sequence, sampleSequence, c)

if __name__ == '__main__':
    main()

