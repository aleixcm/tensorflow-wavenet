import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
#signal = wavfile.read('conSin1.wav')
#signal = signal[1]
#plt.figure(1)
#plt.title('Input Signal - 440 Hz')
#reduced = signal[:440]
#plt.plot(reduced)
#plt.plot(signal)
#plt.show()

#signal2 = wavfile.read('corpus/twoSin/sinus2.wav')
signal2 = wavfile.read('conSin2.wav')
signal2 = signal2[1]
#plt.figure(2)
#plt.title('Input Signal - 440 Hz')
#reduced = signal[:440]
#plt.plot(reduced)
#plt.plot(signal2)


#plot one point
#x_pos = signal2[signal2==1]
#y_pos =
#print(signal2)
plt.figure(1)
NFFT = 1024
Fs = 16000
plt.specgram(signal2, NFFT=NFFT, Fs=Fs, noverlap=900)
#plt.plot(signal2)
plt.show()