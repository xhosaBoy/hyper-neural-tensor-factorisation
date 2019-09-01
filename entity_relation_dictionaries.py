# std
import os
import pickle

def get_path():

    project = os.path.dirname(__file__)
    dirname = 'data/WN18'
    path = os.path.join(project, dirname)

    return path


def save_dictionary(path, filename='wordnet-mlj12-definitions.txt', tuple='entities'):

    words = []
    word2idx = {}

    with open(f'{path}/{filename}', 'rb') as f:

        for l in f:

            line = l.decode().split()
            idx = line[0]

            word = line[1].replace('_', ' ').strip().split()[:-2]
            words.append(word)

            word2idx[idx] = word

    pickle.dump(words, open(f'{path}/{tuple}.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{path}/{tuple}.pkl', 'wb'))


def load_dictionary(get_path, tuple='entities'):

    word = pickle.load(open(f'{get_path}/{tuple}.pkl', 'rb'))
    word2idx = pickle.load(open(f'{get_path}/entity2idx.pkl', 'rb'))

    return word2idx


if __name__ == '__main__':

    save_dictionary(get_path(tuple='entities'))