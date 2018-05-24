import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

signal = wavfile.read('generatedSignals/general/oneSin.wav')
signal = signal[1]
plt.title('Generated Signal - 440 Hz, Q=12800 epoch, 2x1,2,4')
reduced = signal[-440:]
plt.plot(reduced)
plt.show()