# std
import os
import pickle
from datetime import datetime

# 3rd party
import fasttext
import bcolz
import numpy as np


def get_lm_path():

    project = os.path.dirname(__file__)
    dirname = 'data/fasttext'
    path = os.path.join(project, dirname)

    return path


def save_pre_trained_vectors(glove_path):

    words = []
    idx = 0
    err = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.200.dat', mode='w')

    with open(f'{glove_path}/glove.6B.200d.txt', 'rb') as f:

        for l in f:

            line = l.decode().split()
            word = line[0]

            words.append(word)
            word2idx[word] = idx
            idx += 1

            try:
                vect = np.array(line[1:]).astype(np.float)
                if vect.size != 200:
                    vect = np.random.randn(200, ) * np.sqrt(1 / (200 - 1))
            except ValueError as e:
                err += 1
                vect = np.random.randn(200, ) * np.sqrt(1 / (200 - 1))
            finally:
                vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 200)),
                           rootdir=f'{glove_path}/6B.200.dat', mode='w')
    vectors.flush()

    pickle.dump(words, open(f'{glove_path}/6B.200_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.200_idx.pkl', 'wb'))


def load_pre_trained_vectors(glove_path):

    vectors = bcolz.open(f'{glove_path}/6B.200.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.200_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.200_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    return glove


def load_fastext(path):

    model = fasttext.load_model(f'{path}/cc.en.300.bin')

    return model


if __name__ == "__main__":
    print('START:', datetime.now())
    save_pre_trained_vectors(glove_path=get_lm_path())
    glove = load_pre_trained_vectors(get_lm_path())
    print('END:', datetime.now())
