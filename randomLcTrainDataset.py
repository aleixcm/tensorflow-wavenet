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

def createRandomSequence():
    # sequence length
    sq_length = random.randint(5, 10)
    #create sequence
    sequence = []
    sampleSequence = []
    minLen = 1818
    for i in range(0, sq_length):
        value = random.randint(0,2)
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

    file00 = open(os.path.join('corpus', 'localTrainBigDataset_noAmp3', 'lc_train%s.txt' % c), 'w')
    for item in fullSequence:
        file00.write('%i,\n' % item)

    file00.close()

def case(x):
    return {
        0: 440,
        1: 880,
        2: 1320,
    }[x]

def genSignals(sequence, sampleSequence, c):
    y=[]
    for i in range(len(sequence)):
        # convert categories to frequencies
        freq = case(sequence[i])
        nSamples = np.arange(sampleSequence[i])
        #a = random.randint(25, 100)/100
        a = 1
        y0 = a*np.sin(2*np.pi*freq*nSamples / fs)
        y0 = y0.tolist()
        y = y + y0
    noise = 0.01*np.random.normal(0, 1, len(y))
    y = np.asarray(y) + noise
    scipy.io.wavfile.write(os.path.join('corpus', 'localTrainBigDataset_noAmp', 'lc_train%s.wav' % c), fs, y)

def main():
    for c in range(0,100):
        sequence, sampleSequence = createRandomSequence()
        #print(sequence, sampleSequence)
        genFile(sequence, sampleSequence, c)
        genSignals(sequence, sampleSequence, c)

if __name__ == '__main__':
    main()

