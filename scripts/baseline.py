import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import numpy as np
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.models import Model
from keras.layers import Convolution2D, Dense, Input, Flatten, 
                         Dropout, MaxPooling2D, BatchNormalization                         

from sklearn.preprocessing import LabelBinarizer
import keras
from pathlib import Path
import time

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard



#DATADIR = '../data' # unzipped train and test data

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}


def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    
    trainset, valset = load_data(path)
    trainset = pd.DataFrame.from_records(trainset, columns=['word', 'person_id', 'path'])
    valset = pd.DataFrame.from_records(valset, columns=['word', 'person_id', 'path'])

    return trainset[['path', 'word']], valset[['path', 'word']]


def prepare_data(df):
    
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'
    
    return df


def get_specgrams(paths, nsamples=16000):
    '''
    Given list of paths, return specgrams.
    '''
    
    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]

    # zero pad the shorter samples and cut off the long ones.
    data = [] 
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        data.append(d)

    # get the specgram
    specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [s.reshape(129, 124, -1) for s in specgram]
    
    return specgram


def batch_generator(X, y, batch_size=16):
    '''
    Return a random image from X, y
    '''
    
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]
        
        specgram = get_specgrams(im)


        yield np.concatenate([specgram]), label

def get_model(shape):
    '''Create a keras model.'''
    nclass = 11
    inp = Input(shape=shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.binary_crossentropy)
    model.summary()
    
    return model


train, val = get_data('../data')
prepare_data(train)
prepare_data(val)
shape = (129, 124, 1)
model = get_model(shape)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])



# create training and test data.

labelbinarizer = LabelBinarizer()
X = train.path
y = labelbinarizer.fit_transform(train.word)

X_val = val.path
y_val = labelbinarizer.fit_transform(val.word)

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)



train_gen = batch_generator(X.values, y, batch_size=16)
valid_gen = batch_generator(X_val.values, y_val, batch_size=16)

model.fit_generator(
    generator=train_gen,
    epochs=4,
    steps_per_epoch=X.shape[0] // 16,
    validation_data=valid_gen,
    validation_steps=X_val.shape[0] // 16,
    callbacks=[tensorboard])




def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])
    
    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'
    
    return df



test = prepare_data(get_data('../data/test/'))

predictions = []
paths = test.path.tolist()

for path in paths:
    specgram = get_specgrams([path])
    pred = model.predict(np.array(specgram))
    predictions.extend(pred)


labels = [labelbinarizer.inverse_transform(p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
test['labels'] = labels
test.path = test.path.apply(lambda x: str(x).split('/')[-1])
submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
submission.to_csv('submission_1.csv', index=False)