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


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(__file__))
    logger.debug(f'root: {root}')
    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    return path


def save_dictionary(knowledge_graph='wn', attribute='entity', delimiter=' '):
    logger.info(f'Saving {attribute} dictionary ...')
    word2idx = {}

    dirname = 'data/FB15k'
    filename = 'mid2name.tsv'
    path = get_path(filename, dirname)

    with open(path, 'r', encoding='utf-8') as readpickle:
        tsv_reader = csv.reader(readpickle, delimiter=delimiter)

        for line in tsv_reader:
            idx = line[0]
            entity = line[1]
            word2idx[idx] = entity

    filename = f'{knowledge_graph}_{attribute}_map.pkl'
    dirname = 'language_models/FB15k'
    path = get_path(filename, dirname)

    with open(path, 'wb') as writepickle:
        pkl.dump(word2idx, writepickle)

    logger.info(f'Successfully saved {attribute} dictionary!')


def load_dictionary(path):
    logger.info(f'Loading entity dictionary ...')
    with open(path, 'rb') as readpickle:
        word2idx = pkl.load(readpickle)

    logger.info(f'Successfully loaded entity dictionary!')
    return word2idx


if __name__ == '__main__':
    logger.info('START!')
    knowledge_graph = 'fb15k'
    attribute = 'entity'

    save_dictionary(knowledge_graph, attribute, '\t')

    filename = f'{knowledge_graph}_{attribute}_map.pkl'
    dirname = 'language_models/FB15k'
    path = get_path(filename, dirname)
    fb_entity_map = load_dictionary(path)
    logger.info('DONE!')
