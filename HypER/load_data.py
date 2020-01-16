import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Data:

    def __init__(self, data_dir="data/FB15k-237", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.data_train_and_valid = self.train_data + self.valid_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations \
                         + [i for i in self.valid_relations if i not in self.train_relations] \
                         + [i for i in self.test_relations if i not in self.train_relations]

    @staticmethod
    def load_data(dirname, data_type="train", reverse=False):
        root, _ = os.path.split(os.path.dirname(__file__))
        logger.debug(f'root: {root}')
        filename = f'{data_type}.txt'
        path = os.path.join(root, dirname, filename)
        logger.debug(f'path: {path}')

        with open(path, 'r') as f:
            data = f.read().strip().split('\n')
            data = [i.split() for i in data]

            if reverse:
                data += [[i[2], f'{i[1]}_reverse', i[0]] for i in data]

        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
