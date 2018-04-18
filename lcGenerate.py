import os
import scipy.io.wavfile

fs = 16000

# Categories
a = [0]
b = [1]
c = [2]

# Samples per Category
aSamples = 8000
bSamples = 8000
cSamples = 8000

def main():
    sequence = a *aSamples + b*bSamples
    file00 = open(os.path.join('corpus','local','lc_gen0.txt'), 'w')
    for item in sequence:
        file00.write("%s\n" % item)

    file00.close()

if __name__ == '__main__':
    main()