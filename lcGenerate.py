import os
import scipy.io.wavfile

fs = 16000

# Categories
a = [0]
b = [1]
c = [2]

# Samples per Category
aSamples = 24000
bSamples = 24000
cSamples = 24000
def main():
    sequence = a *aSamples + b*bSamples + c*cSamples
    file00 = open(os.path.join('corpus','localTrain','lc_gen4_72000.txt'), 'w')
    for item in sequence:
        file00.write("%s,\n" % item)

    file00.close()

if __name__ == '__main__':
    main()
