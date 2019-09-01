from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# std
import sys
import logging
import argparse
from collections import defaultdict

# 3rd party
import torch
import numpy as np
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

# internal
from load_data import Data
from models import *
from pre_trained_word_vectors import *
from entity_relation_dictionaries import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('train.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Experiment:

    def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=300, rel_vec_dim=300,
                 num_iterations=100, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

        self.loss = torch.nn.BCELoss()

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):

        batch = er_vocab_pairs[idx:min(idx + self.batch_size, len(er_vocab_pairs))]

        targets = np.zeros((len(batch), len(d.entities)))

        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.

        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()

        return np.array(batch), targets

    def evaluate(self, model, data):

        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        val_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(val_data_idxs))

        for i in range(0, len(val_data_idxs), self.batch_size):

            data_batch, _ = self.get_batch(er_vocab, val_data_idxs, i)

            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            logits = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):

                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = logits[j, e2_idx[j]].item()
                logits[j, filt] = 0.0
                logits[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(logits, dim=1, descending=True)

            for j in range(data_batch.shape[0]):

                rank = np.where(sort_idxs[j].cpu() == e2_idx[j].cpu())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        indexes = [(index + 1) for index in e2_idx[range(5)]]
        print(f'random predictions {logits[[range(5)], indexes]}')
        print(f'target predictions: {logits[[range(5)], e2_idx[range(5)]]}')

        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self, entity2idx, language_model):

        logger.info("Training the %s model..." % model_name)
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}

        matrix_entity_len = len(d.entities)
        print(f'matrix_entity_len: {matrix_entity_len}')
        weights_entity_matrix = np.zeros((matrix_entity_len, 300))

        entities_found = 0

        for entity_idx in self.entity_idxs.keys():

            i = self.entity_idxs[entity_idx]

            entity_found = False
            embedding = []

            try:
                entity_string = entity2idx[str(entity_idx)]
                for entity in entity_string:
                    embedding.append(language_model[entity])
                    entity_found = True
                weights_entity_matrix[i] = np.array(embedding).mean(axis=0)
            except KeyError:
                if not embedding:
                    # weights_entity_matrix[i] = np.random.normal(scale=0.6, size=(300,))
                    weights_entity_matrix[i] = np.random.randn(300,) * np.sqrt(1 / (300 - 1))
                else:
                    weights_entity_matrix[i] = np.array(embedding).mean(axis=0)
            finally:
                if entity_found:
                    entities_found += 1

        print(f'entities_found: {entities_found}')
        print(f'entity ids: {len(self.entity_idxs.keys())}')
        print(f'pre trained coverage: {(entities_found / len(self.entity_idxs.keys()) * 100):.2f}%')
        self.entity_weights = weights_entity_matrix
        print(f'weights_entity_matrix size: {weights_entity_matrix.size}')

        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        relation2idx = {str(i): d.relations[i].replace(
            '_', ' ').strip().split() for i in range(len(d.relations))}

        matrix_relation_len = len(d.relations)
        print(f'matrix_relation_len: {matrix_relation_len}')
        weights_relation_matrix = np.zeros((matrix_relation_len, 300))
        relations_found = 0

        for relation_idx in self.relation_idxs.keys():

            i = self.relation_idxs[relation_idx]

            embedding = []
            relation_found = False

            try:
                relation_string = relation2idx[str(i)]
                for relation in relation_string:
                    embedding.append(language_model[relation])
                    relation_found = True
                weights_relation_matrix[i] = np.array(embedding).mean(axis=0)
            except KeyError:
                if not embedding:
                    # weights_relation_matrix[i] = np.random.normal(scale=0.6, size=(300,))
                    weights_entity_matrix[i] = np.random.randn(300, ) * np.sqrt(1 / (300 - 1))
                else:
                    weights_relation_matrix[i] = np.array(embedding).mean(axis=0)
            finally:
                if relation_found:
                    relations_found += 1

        print(f'relations_found: {relations_found}')
        self.relation_weights = weights_relation_matrix
        print(f'weights_relation_matrix size: {weights_relation_matrix.size}')

        train_data_idxs = self.get_data_idxs(d.train_data)
        logger.info("Number of training data points: %d" % len(train_data_idxs))

        if model_name.lower() == "hype":
            model = HypE(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "hyper":
            model = HypER(d, self.ent_vec_dim, self.rel_vec_dim,
                          weights_entity_matrix, weights_relation_matrix, **self.kwargs)
        elif model_name.lower() == "distmult":
            model = DistMult(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "conve":
            model = ConvE(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "complex":
            model = ComplEx(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        print([value.numel() for value in model.parameters()])

        if self.cuda:
            model.cuda()

        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        print(len(er_vocab_pairs))

        print("Starting training...")

        for epoch in range(1, self.num_iterations + 1):

            logger.info(f'EPOCH: {epoch}')

            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            iteration = 0
            for j in range(0, len(er_vocab_pairs), self.batch_size):

                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])

                logger.debug(f'targets size: {targets.size()}')

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()

                # logits = model.forward(e1_idx, r_idx)
                # logger.debug(f'logits size: {logits.size()}')
                # predictions = F.sigmoid(logits)
                predictions = model.forward(e1_idx, r_idx)
                logger.debug(f'logits size: {predictions.size()}')

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss = self.loss(predictions, targets)
                accuracy = model.accuracy(predictions, targets).item()

                # if j % (self.batch_size * 10) == 0:
                    # logger.info(f'ITERATION: {iteration + 1}')
                    # logger.info(f'loss: {loss}')
                    # logger.info(f'accuracy: {accuracy}')

                loss.backward()
                opt.step()

                iteration += 1

            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            logger.info(f'EPOCH: {epoch}')
            logger.info(f'Loss: {np.mean(losses)}')

            model.eval()
            with torch.no_grad():
                logger.info("Validation:")
                self.evaluate(model, d.valid_data)
                if (epoch % 10) == 0:
                    logger.info("Test:")
                    self.evaluate(model, d.test_data)

        model.eval()
        print("Test:")
        self.evaluate(model, d.test_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="HypER", nargs="?",
                        help='Which algorithm to use: HypER, ConvE, DistMult, or ComplEx')
    parser.add_argument('--dataset', type=str, default="WN18", nargs="?",
                        help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    args = parser.parse_args()

    model_name = args.algorithm

    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    d = Data(data_dir=data_dir, reverse=True)

    torch.backends.cudnn.deterministic = True
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    experiment = Experiment(model_name, num_iterations=200, batch_size=128, learning_rate=0.001,
                            decay_rate=0.99, ent_vec_dim=300, rel_vec_dim=300, cuda=False,
                            input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2,
                            in_channels=1, out_channels=32, filt_h=1, filt_w=9, label_smoothing=0.1)

    entity2idx = load_dictionary(get_path(), tuple='entities')
    # relation2idx = load_dictionary(get_path(), tuple='relations')

    # language_model = load_pre_trained_vectors(get_glove_path())
    language_model = load_fastext(get_lm_path())

    experiment.train_and_eval(entity2idx, language_model)
