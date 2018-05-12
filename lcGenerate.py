import os
import scipy.io.wavfile

fs = 16000

# Categories
a = [0]
b = [1]
c = [2]
d = [3]
e = [4]
f = [5]
g = [6]

# Samples per Category
aSamples = 8000
bSamples = 8000
cSamples = 8000
dSamples = 8000
eSamples = 8000
fSamples = 8000
gSamples = 8000

def main():
    sequence = a *aSamples + b*bSamples + c*cSamples + d*dSamples + e*eSamples + f*fSamples +g*gSamples
    file00 = open(os.path.join('corpus','Analysis','7freq_56000.txt'), 'w')
    for item in sequence:
        file00.write("%s,\n" % item)

    file00.close()

if __name__ == '__main__':
    main()
