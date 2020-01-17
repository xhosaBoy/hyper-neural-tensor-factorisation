# std
import os
import sys
import pickle
import logging
from datetime import datetime

# 3rd party
import fasttext
import bcolz
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, dirname, filename) if filename else os.path.join(root, filename)
    return path


def save_language_model(language_model_data, language_model_version, dirname=None):
    logger.info(f'Saving Glove language model version {language_model_version} ...')

    words = []
    word2idx = {}
    err = 0

    language_model_file = f'{language_model_version}.dat'
    language_model = get_path(language_model_file, dirname)
    vectors = bcolz.carray(np.zeros(1), rootdir=language_model, mode='w')

    path = get_path(language_model_data, dirname)
    with open(path, 'rb') as f:
        for idx, l in enumerate(f):
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx

            try:
                vect = np.array(line[1:]).astype(np.float)
                if vect.size != 200:
                    vect = np.random.randn(200) * np.sqrt(1 / (200 - 1))
            except ValueError as e:
                err += 1
                vect = np.random.randn(200) * np.sqrt(1 / (200 - 1))
            finally:
                vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 200)), rootdir=language_model, mode='w')
    vectors.flush()

    filename = f'{language_model_version}_words.pkl'
    path = get_path(filename, dirname)

    with open(path, 'wb') as language_model_words:
        pickle.dump(words, language_model_words)

    filename = f'{language_model_version}_idx.pkl'
    path = get_path(filename, dirname)

    with open(path, 'wb') as language_model_ids:
        pickle.dump(word2idx, language_model_ids)

    logger.info(f'Saving Glove language model version {language_model_version} complete!')


def load_glove(language_model_version, dirname=None):
    logger.info(f'Loading Glove language model ...')

    filename = f'{language_model_version}.dat'
    path = get_path(filename, dirname)

    vectors = bcolz.open(path)[:]

    filename = f'{language_model_version}_words.pkl'
    path = get_path(filename, dirname)

    with open(path, 'rb') as wordsfile:
        words = pickle.load(wordsfile)

    filename = f'{language_model_version}_idx.pkl'
    path = get_path(filename, dirname)

    with open(path, 'rb') as word_ids:
        word2idx = pickle.load(word_ids)

    glove = {w: vectors[word2idx[w]] for w in words}

    logger.info(f'Loading Glove language model complete!')

    return glove


def load_fastext():
    logger.info(f'Loading Fasttext language model ...')

    language_model_name = 'cc.en.300.bin'
    dirname = 'language_models/fasttext'
    path = get_path(language_model_name, dirname)

    language_model = fasttext.load_model(path)

    logger.info(f'Loading Fasttext language model complete!')

    return language_model


if __name__ == "__main__":
    logger.info('START!')

    language_model_data = 'glove.6B.200d.txt'
    language_model_version = '6B.200'
    dirname = 'language_models/glove'

    save_language_model(language_model_data, language_model_version, dirname)
    glove = load_glove(language_model_version, dirname)

    logger.info('DONE!')
