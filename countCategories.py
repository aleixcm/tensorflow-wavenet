import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa.display
from sklearn import preprocessing
import scipy.io.wavfile as wavfile
import os

def read_category_id_local(labelsFileName):
    with open(labelsFileName, 'r') as myfile:
        category_id_local = myfile.read().replace('\n', '')
    return(category_id_local)

def get_category_cardinality(cardinality, category_id_local):
    for item in category_id_local:
        if item > cardinality:
            cardinality = item

    return(cardinality)

def counter(total, cardinalityArray, category_id_local):
    for item in category_id_local:
        i = 0
        while i <= len(cardinalityArray):
            if item == cardinalityArray[i]:
                total[i] = total[i]+1
                break
            else:
                i += 1
    return (total)

def main():
    path = os.path.join('corpus','prova')
    cardinality = 0
    for file in os.listdir(path):
        if file.endswith(".txt"):
            fileName = os.path.join(path,file)
            labels = read_category_id_local(fileName)
            category_id_local = np.fromstring(labels, dtype=int, sep=',').reshape(-1, 1)
            cardinality = get_category_cardinality(cardinality, category_id_local)

    cardinality = cardinality+1
    cardinalityArray = np.arange(cardinality)
    total = np.zeros(cardinality)

    for file in os.listdir(path):
        if file.endswith(".txt"):
            fileName = os.path.join(path, file)
            labels = read_category_id_local(fileName)
            category_id_local = np.fromstring(labels, dtype=int, sep=',').reshape(-1, 1)
            total = counter(total, cardinalityArray, category_id_local)

    print(total)

if __name__ == '__main__':
    main()