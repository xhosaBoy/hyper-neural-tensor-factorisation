# std
import sys
import random
import logging
from collections import defaultdict

# 3rd party
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

# internal
from models import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


class Experiment:

    def __init__(self, model_name, d, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=100, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
        self.model_name = model_name
        self.d = d
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

        # set all e2 relations for e1,r pair to true
        targets = np.zeros((len(batch), len(self.d.entities)))

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

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.d.data))
        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):

            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):

                # Set ojbect predictions not in batch to zero
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()

            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):

        print("Training the %s model..." % self.model_name)
        self.entity_idxs = {self.d.entities[
            i]: i for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[
            i]: i for i in range(len(self.d.relations))}
        train_data_idxs = self.get_data_idxs(self.d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model_name.lower() == "hype":
            model = HypE(self.d, self.ent_vec_dim,
                         self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "hyper":
            model = HypER(self.d, self.ent_vec_dim,
                          self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "distmult":
            model = DistMult(self.d, self.ent_vec_dim,
                             self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "conve":
            model = ConvE(self.d, self.ent_vec_dim,
                          self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "complex":
            model = ComplEx(self.d, self.ent_vec_dim,
                            self.rel_vec_dim, **self.kwargs)
        print([value.numel() for value in model.parameters()])

        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        view_key = list(er_vocab.keys())[0]
        print('sample enitity:', view_key)

        er_vocab_pairs = list(er_vocab.keys())
        print(len(er_vocab_pairs))

        print("Starting training...")

        for it in range(1, self.num_iterations + 1):

            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            # for j in range(0, len(er_vocab_pairs), self.batch_size):
            for j in range(0, (self.batch_size * 1), self.batch_size):

                if j % (self.batch_size * 100) == 0:
                    logger.info(f'ITERATION: {j + 1}')
                data_batch, targets = self.get_batch(
                    er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()

                predictions = model.forward(e1_idx, r_idx)

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) *
                               targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)

                if j % (self.batch_size * 10) == 0:
                    logger.info(f'loss: {loss}')

                loss.backward()
                opt.step()

            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            print(it)
            print(np.mean(losses))

            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, self.d.valid_data)
                if not it % 2:
                    print("Test:")
                    self.evaluate(model, self.d.test_data)


class ExperimentProxE:

    def __init__(self, model_name, d, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
                 num_epoch=100, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 in_channels=1, out_channels=32, filt_h=3, filt_w=3, label_smoothing=0.):
        self.model_name = model_name
        self.d = d
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout": hidden_dropout,
                       "feature_map_dropout": feature_map_dropout, "in_channels": in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

    def get_data_idxs(self, data):

        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]

        return data_idxs

    def get_sp_vocab(self, data):

        sp_vocab = defaultdict(list)
        for triple in data:
            sp_vocab[(triple[0], triple[1])].append(triple[2])

        return sp_vocab

    def get_po_vocab(self, data):

        po_vocab = defaultdict(list)
        for triple in data:
            po_vocab[(triple[1], triple[2])].append(triple[0])

        return po_vocab

    def get_batch(self, train_data_idxs, sp_vocab, sp_vocab_pairs, po_vocab_pairs, idx):

        spo_batch = train_data_idxs[idx:min(idx + self.batch_size, len(train_data_idxs))]
        sp_batch = [(triple[0], triple[1]) for triple in spo_batch]

        # sample random value an entity is part of
        e2 = defaultdict(list)
        for pair in po_vocab_pairs:
            e2[pair[1]].append(pair[0])

        po_batch = [(random.choice(predicates), entity) for entity, predicates in e2.items()]

        # build target: set all e2 relations for e1,r pair to true, binary loss
        # at first
        targets = np.zeros((len(sp_batch), len(po_batch)))

        for idx, pair in enumerate(sp_batch):
            targets[idx, sp_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)

        if self.cuda:
            targets = targets.cuda()

        return np.array(spo_batch), np.array(po_batch), targets

    def evaluate(self, model, data, po_vocab_pairs):

        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sp_vocab = self.get_sp_vocab(self.get_data_idxs(self.d.data))
        sp_vocab_pairs = list(sp_vocab.keys())

        logger.info("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):

            data_batch, po_batch, _ = self.get_batch(test_data_idxs, sp_vocab, sp_vocab_pairs, po_vocab_pairs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2b_idx = torch.tensor(data_batch[:, 2])
            r2_idx = torch.tensor(po_batch[:, 0])
            e2_idx = torch.tensor(po_batch[:, 1])

            logger.debug(f'e2_idx[0]: {e2_idx[0]}')

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2b_idx = e2b_idx.cuda()
                r2_idx = r2_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions = model.forward(e1_idx, r_idx, r2_idx, e2_idx)

            for j in range(data_batch.shape[0]):

                # Set ojbect predictions not in batch to zero
                filt = sp_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2b_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2b_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()

            logger.debug(f'sorted object indecies: {sort_idxs[0]}')

            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2b_idx[j])[0][0]
                ranks.append(rank + 1)

                logger.debug(f'object ranks for batch: {rank}')

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):

        # map entities, relations, and training data to ids
        logger.info('Training the %s model...' % self.model_name)
        self.entity_idxs = {self.d.entities[
            i]: i for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[
            i]: i for i in range(len(self.d.relations))}
        train_data_idxs = self.get_data_idxs(self.d.train_data)
        logger.info('Number of training data points: %d' %
                    len(train_data_idxs))

        if self.model_name.lower() == "proxe":
            model = ProxE(self.d, self.ent_vec_dim,
                          self.rel_vec_dim, self.batch_size, **self.kwargs)
        elif self.model_name.lower() == "hype":
            model = HypE(self.d, self.ent_vec_dim,
                         self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "hyper":
            model = HypER(self.d, self.ent_vec_dim,
                          self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "distmult":
            model = DistMult(self.d, self.ent_vec_dim,
                             self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "conve":
            model = ConvE(self.d, self.ent_vec_dim,
                          self.rel_vec_dim, **self.kwargs)
        elif self.model_name.lower() == "complex":
            model = ComplEx(self.d, self.ent_vec_dim,
                            self.rel_vec_dim, **self.kwargs)

        logger.info(f'Model parameters: {[value.numel() for value in model.parameters()]}')

        if self.cuda:
            model.cuda()

        model.init()

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        sp_vocab = self.get_sp_vocab(train_data_idxs)
        sp_vocab_pairs = list(sp_vocab.keys())

        logger.debug(f'sample ER: {sp_vocab_pairs[0]}')
        logger.debug(f'predicate sample: {sp_vocab[sp_vocab_pairs[0]]}')
        logger.debug(f'subject object pair count: {len(sp_vocab_pairs)}')
        logger.debug(f'train_data_idxs: {train_data_idxs[:2]} ... {train_data_idxs[-2:]}')

        logger.info('Starting training...')

        po_vocab = self.get_po_vocab(train_data_idxs)
        po_vocab_pairs = list(po_vocab.keys())

        for epoch in range(1, self.num_epoch + 1):

            logger.info(f'EPOCH: {epoch}')

            model.train()
            losses = []
            np.random.shuffle(train_data_idxs)

            for j in range(0, len(sp_vocab_pairs), self.batch_size):

                if j % (self.batch_size * 100) == 0:
                    logger.info(f'ITERATION: {j + 1}')
                spo_batch, po_batch, targets = self.get_batch(
                    train_data_idxs, sp_vocab, sp_vocab_pairs, po_vocab_pairs, j)
                opt.zero_grad()

                e1_idx = torch.tensor(spo_batch[:, 0])
                r_idx = torch.tensor(spo_batch[:, 1])
                r2_idx = torch.tensor(po_batch[:, 0])
                e2_idx = torch.tensor(po_batch[:, 1])

                logger.debug(f'e2 size: {e2_idx.size()}')
                logger.debug(f'targets size: {targets.size()}')
                logger.debug(f'target classes: {torch.max(targets[0], 0)[1]}')

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    r2_idx = r2_idx.cuda()
                    e2_idx = e2_idx.cuda()

                predictions = model.forward(e1_idx, r_idx, r2_idx, e2_idx)
                logger.debug(f'preditions size: {predictions.size()}')

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) *
                               targets) + (1.0 / targets.size(1))

                loss = model.loss(predictions, targets)
                accuracy = model.accuracy(predictions, targets).item()

                if j % (self.batch_size * 10) == 0:
                    logger.info(f'loss: {loss}')
                    logger.info(f'accuracy: {accuracy}')

                loss.backward()
                opt.step()

            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            logger.info(f'Mean losses: {np.mean(losses)}')
            logger.info(f'VALIDATION: {epoch}')

            model.eval()
            with torch.no_grad():
                self.evaluate(model, self.d.valid_data, po_vocab_pairs)
                # if not epoch % 2:
                #     print("Test:")
                #     self.evaluate(model, self.d.test_data)
