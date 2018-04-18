import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.io.wavfile
import os

fs = 16000
t = 16000        #Time in samples
t = np.arange(t)

def sinus(f0):
    y = np.sin(2 * np.pi*f0*t/fs)
    return y

def square(f0):
    y = signal.square(2 * np.pi * f0 * t/fs)
    return y

def sawtooth(f0):
    y = signal.sawtooth(2 * np.pi * f0 * t / fs)
    return y

def main():
    f0 = [440,880]
    a = 1
    for i in range(0,2):
        ySin = a * sinus(f0[i])
        ySquare = a * square(f0[i])
        ySinName = '00' + 'signal' + '0'+ str(i) + '.wav'
        ySquareName = '01' + 'signal' + '0'+ str(i) + '.wav'
        scipy.io.wavfile.write(os.path.join('corpus','localGlobal', ySinName), fs, ySin)
        scipy.io.wavfile.write(os.path.join('corpus','localGlobal', ySquareName), fs, ySquare)

if __name__ == '__main__':
    main()