import os
import scipy.io.wavfile

fs = 16000

# Categories
a = [0]
b = [1]
c = [2]

# Samples per Category
aSamples = 16000
bSamples = 16000
cSamples = 16000
def main():
    sequence = c *cSamples + b*bSamples + a*aSamples
    file00 = open(os.path.join('corpus','localTrain','lc_gen3_48000.txt'), 'w')
    for item in sequence:
        file00.write("%s,\n" % item)

    file00.close()

if __name__ == '__main__':
    main()
