import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa.display
from sklearn import preprocessing
import scipy.io.wavfile as wavfile
import os
import codecs
import json

#np.set_printoptions(threshold=np.nan)
y, sr = librosa.load('corpus/Analysis/lc_scale_flute.wav', sr=16000)

def calculateMFCC(y,sr):
    mfccs = librosa.feature.mfcc(y, sr=sr)
    mfccs = preprocessing.scale(mfccs, axis=1)
    return(mfccs)

def Upsampling(labels):
    padding = int(len(y)/len(labels)/2)
    upLabels = []
    for i, item in enumerate(labels):
        padLabel = np.pad([labels[i]], (padding, padding-1), 'constant', constant_values=item)
        upLabels = np.append(upLabels, padLabel)
    # Fix lenghts
    if len(upLabels) > len(y):
        upLabels = upLabels[:len(y)]
    elif len(upLabels) < len(y):
        diff = len(y)-len(upLabels)
        upLabels = np.pad(upLabels, (0, diff), 'constant', constant_values=upLabels[len(upLabels)-1])
    return upLabels

def genFile(upLabels12):
    '''
    a = np.arange(10).reshape(2, 5)  # a 2 by 5 array
    b = a.tolist()  # nested lists with same data, indices
    file_path = "/path.json"  ## your path variable
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    '''

    upLabelsJSON = upLabels12.tolist()
    file_path = os.path.join('corpus', 'Analysis', 'lc_scale_flute.json')
    json.dump(upLabelsJSON, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)
    return (file_path)

'''
    file00 = open(os.path.join('corpus', 'Analysis', 'mfcc.json'), 'w')
    for i in range(upLabels12.shape[0]):
        for item in (upLabels12[i][:]):
            file00.write('%f,' % item)
        file00.write(';')
    file00.close()
    return file00.name
'''


def readFile(fileName):
    file_path = os.path.join('corpus', 'Analysis', 'mfcc.json')
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    upLabelsJSONnew = json.loads(obj_text)
    upLabelsNew = np.array(upLabelsJSONnew)
    return(upLabelsNew)

    '''
    with open(fileName, 'r') as myfile:
        readLabels = myfile.read()
        #convert into 12 channels again
        readLabels2=readLabels.split(';')
        readLabels2=readLabels2[:12]
        matrix = np.fromstring(readLabels2[0], dtype=float, sep=',').reshape(-1, 1)  # np.array
        for i in range(1,12):
            row = np.fromstring(readLabels2[i], dtype=float, sep=',').reshape(-1, 1)  # np.array
            matrix = np.hstack((matrix, row))
    return(matrix)
    '''

def main():
    mfccs=calculateMFCC(y,sr)
    upLabels12 = Upsampling(mfccs[1][:])
    for i in range(2,13):
        upLabels=Upsampling(mfccs[i][:])
        upLabels12 = np.vstack((upLabels12, upLabels))

    fileName=genFile(upLabels12)
    #upLabelsNew = readFile(fileName)
    #print('a')

    '''
    plt.figure()
    for i in range(0,12):

        plt.plot(upLabels12[i][:])

    plt.show()
    #plots
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='mel')
    # librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('MFCCs')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    '''
if __name__ == '__main__':
    main()