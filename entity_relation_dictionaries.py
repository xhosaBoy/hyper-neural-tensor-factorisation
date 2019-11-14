# std
import os
import sys
import csv
import logging
import pickle as pkl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(dirname, filename=None):

    project = os.path.dirname(__file__)
    path = os.path.join(project, dirname, filename) if filename else os.path.join(project, dirname)

    return path


def save_dictionary(readpath, kg='wn', element='entity', delimiter=' '):

    logger.info(f'Saving {element} dictionary...')

    word2idx = {}

    with open(f'{readpath}', 'r', encoding='utf-8') as readpickle:

        tsv_reader = csv.reader(readpickle, delimiter=delimiter)
        for line in tsv_reader:
            idx = line[0]
            entity = line[1]
            word2idx[idx] = entity

    writepath = get_path('data/FB15k')
    with open(f'{writepath}/{kg}_{element}_map.pkl', 'wb') as writepickle:
        pkl.dump(word2idx, writepickle)

    logger.info(f'Successfully saved {element} dictionary!')


def load_dictionary(path, kg, element='entity'):

    with open(f'{path}/{kg}_{element}_map.pkl', 'rb') as readpickle:
        word2idx = pkl.load(readpickle)

    return word2idx


if __name__ == '__main__':

    dirname = 'data/FB15k'
    filename = 'mid2name.tsv'
    kg = 'fb'
    element = 'entity'

    path = get_path(dirname, filename)
    save_dictionary(path, kg, element, '\t')
    path = get_path(dirname)
    fb_entity_map = load_dictionary(path, kg, element)
