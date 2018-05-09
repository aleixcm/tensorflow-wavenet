import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

from sklearn import preprocessing
import codecs
import json

#FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'          #VCTK Corpus
#FILE_PATTERN = r'sinus([0-9])\.wav'                #scale
#FILE_PATTERN = r'([0-9]+)cat([0-9]+)\.wav'         #shape
#FILE_PATTERN = r'([0-9])+signal+([0-9])\.wav'      #local, localGLobal

FILE_PATTERN = r'lc_train+([0-9]+)\.wav'            #localTrain

'''
#ibab
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id
'''

'''
#aleix 22/03/2018
#for twoSinOctave, scale, fourAmp etc.
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id = int(matches)
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id
#aleix 22/03/2018
'''
'''
#aleix
#for localGlobal, etc
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    min_id_local = None
    max_id_local = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        #id, recording_id = [int(id_) for id_ in matches]
        id, id_local = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

        if min_id_local is None or id_local < min_id_local:
            min_id_local = id_local
        if max_id_local is None or id_local > max_id_local:
            max_id_local = id_local

    return min_id, max_id, min_id_local, max_id_local
#aleix
'''
'''
#aleix
#for localTrain
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    min_id_local = None
    max_id_local = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        #recording_id, id = [int(id_) for id_ in matches]   #for shape use this
        #id = int(matches)                                  #scale use this
        id = [int(id_) for id_ in matches]                 #for localTrain
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id
        labelsFileName = re.sub('\.wav$', '', filename)+'.txt'
        if os.path.isfile(labelsFileName) is not None:
            labels = read_category_id_local(labelsFileName) #str
            labels = np.fromstring(labels, dtype=int, sep=',')
            id_local = np.unique(labels)
            for i in id_local:
                if min_id_local is None or id_local[i] < min_id_local:
                    min_id_local = id_local[i]
                if max_id_local is None or id_local[i] > max_id_local:
                    max_id_local = id_local[i]
        else:
            id_local = None

    return min_id, max_id, min_id_local, max_id_local

#aleix
'''
'''
#aleix
#for localTrainBigDataset
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    min_id_local = None
    max_id_local = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        #recording_id, id = [int(id_) for id_ in matches]   #for shape use this
        #id = int(matches)                                  #scale use this
        id = [int(id_) for id_ in matches]                 #for localTrain
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id
        labelsFileName = re.sub('\.wav$', '', filename)+'.txt'
        if os.path.isfile(labelsFileName) is not None:
            labels = read_category_id_local(labelsFileName) #str
            labels = np.fromstring(labels, dtype=int, sep=',')
            id_local = np.unique(labels)
            for i in range(len(id_local)):
                if min_id_local is None or id_local[i] < min_id_local:
                    min_id_local = id_local[i]
                if max_id_local is None or id_local[i] > max_id_local:
                    max_id_local = id_local[i]
        else:
            id_local = None

    return min_id, max_id, min_id_local, max_id_local

#aleix
'''
#Calculate MFCC. Called by get_labels
def calculateMFCC(y,sr):
    mfccs = librosa.feature.mfcc(y, sr=sr)
    mfccs = preprocessing.scale(mfccs, axis=1)
    return(mfccs)

#Upsampling after MFCC. Called by get_labels
def Upsampling(labels, y):
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

#Generate a JSON File. Called by get_labels
def genFile(upLabels12, fileNameJSON):
    '''
    a = np.arange(10).reshape(2, 5)  # a 2 by 5 array
    b = a.tolist()  # nested lists with same data, indices
    file_path = "/path.json"  ## your path variable
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    '''

    upLabelsJSON = upLabels12.tolist()
    #file_path = os.path.join('corpus', 'Analysis', 'mfcc.json')
    json.dump(upLabelsJSON, codecs.open(fileNameJSON, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
              indent=4)

#aleix
#Get labels
def get_labels(filename):
    y, sr = librosa.load(filename, sr=16000)
    mfccs = calculateMFCC(y, sr)
    upLabels12 = Upsampling(mfccs[1][:],y)
    for i in range(2, 13):
        upLabels = Upsampling(mfccs[i][:],y)
        upLabels12 = np.vstack((upLabels12, upLabels))

    #Create JSON Name
    base = os.path.splitext(filename)[0]
    fileNameJSON = base + '.json'

    genFile(upLabels12, fileNameJSON)


#for localTrainBigDataset
def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    min_id_local = None
    max_id_local = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        #recording_id, id = [int(id_) for id_ in matches]   #for shape use this
        #id = int(matches)                                  #scale use this
        id = [int(id_) for id_ in matches]                  #for localTrain
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id
        #create labelsFileName
        get_labels(filename)

    return min_id, max_id

#aleix

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

#aleix
def read_category_id_local(labelsFileName):
    obj_text = codecs.open(labelsFileName, 'r', encoding='utf-8').read()
    upLabelsJSONnew = json.loads(obj_text)
    upLabelsNew = np.array(upLabelsJSONnew)
    return (upLabelsNew)
#aleix


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)                     #only.wav
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        labelsFileName = re.sub('\.wav$', '', filename)+'.json'
        if os.path.isfile(labelsFileName):
            labels = read_category_id_local(labelsFileName) #str
            category_id_local = np.transpose(labels)
        else:
            category_id_local = None

        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0]) # only for global
            #category_id, category_id_local = [int(ids[0][0]),int(ids[0][1])] #global and local on name
            #aleix

        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id, category_id_local


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    #print(id_reg_exp)
    for file in files:
        #print(file)
        ids = id_reg_exp.findall(file)
        #print(ids)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 lc_channels,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.lc_channels = lc_channels
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])
        '''
        if self.lc_channels:
            self.id_placeholder_lc = tf.placeholder(dtype=tf.float32, shape=(None, self.lc_channels))
            self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                                shapes=[(None, self.lc_channels)])
            self.lc_enqueue = self.lc_queue.enqueue([self.id_placeholder_lc])
        '''
        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Not needed
        #if self.lc_channels and not_all_have_id(files):
            #raise ValueError("Local conditioning is enabled, but file names "
                             #"do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

        if self.lc_channels:
            _, _ = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.lc_category_cardinality = True
            #print("Detected --lc_cardinality={}".format(
            #      self.lc_category_cardinality))
            # Calculate lc_channels needed
            self.lc_channels = 12
            print("Created --lc_channels={}".format(
                self.lc_channels))

        else:
            self.lc_category_cardinality = None

        if self.lc_channels:
            self.id_placeholder_lc = tf.placeholder(dtype=tf.float32, shape=(None, self.lc_channels))
            self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                                shapes=[(None, self.lc_channels)])
            self.lc_enqueue = self.lc_queue.enqueue([self.id_placeholder_lc])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def dequeue_lc(self, num_elements):
        return self.lc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, filename, category_id, category_id_local in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')

                if category_id_local is not None:
                    # Adding 0 padding of receptive field size at the beggining. Because of causal convolution
                    category_id_local = np.pad(category_id_local, [[self.receptive_field, 0], [0, 0]],
                                   'constant')

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                        if self.lc_channels:

                            category_id_local_piece = category_id_local[:(self.receptive_field +
                                                                          self.sample_size), :]

                            sess.run(self.lc_enqueue, feed_dict={
                                self.id_placeholder_lc: category_id_local_piece})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})
                    if self.lc_channels:
                        sess.run(self.lc_enqueue,
                                 feed_dict={self.id_placeholder_lc: category_id_local})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
