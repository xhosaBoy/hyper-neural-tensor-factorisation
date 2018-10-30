# std
import time
import copy
import pickle
import random
import argparse
from collections import defaultdict
from pprint import pprint

# 3rd party
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.optim.lr_scheduler import ExponentialLR

# internal
from load_data import Data
from models import *


class Experiment:

    def __init__(self, model_name, learning_rate=0.001, ent_vec_dim=200, rel_vec_dim=200,
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
                       "feature_map_dropout": feature_map_dropout, "in_channels":in_channels,
                       "out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w}

    def get_data_idxs(self, data):

        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]

        return data_idxs

    def get_er_vocab(self, data):

        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])

        return er_vocab

    def get_batch(self, train_data_idxs, er_vocab, er_vocab_pairs, idx):

        batch = list()
        # batch = train_data_idxs[idx:min(idx + self.batch_size, len(train_data_idxs))]
        targets = np.zeros((len(batch), len(d.entities))) # set all e2 relations for e1,r pair to true
        corrupt_keys = copy.deepcopy(er_vocab_pairs)

        for i in range(idx, min(idx + self.batch_size, len(train_data_idxs))):
            key = (train_data_idxs[i][0],train_data_idxs[i][1])
            if key in corrupt_keys:
                corrupt_keys.remove(key)
            corrupt_key = random.choice(corrupt_keys)
            e_corrupt = er_vocab[corrupt_key][0]
            training_sample = train_data_idxs[i][0], train_data_idxs[i][1], train_data_idxs[i][2], e_corrupt
            batch.append(training_sample)

        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()

        # return np.array(batch), targets
        return np.array(batch), targets


    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j])[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))


    def train_and_eval(self):

        # map entities, relations, and training data to ids
        print('Training the %s model...' % model_name)
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        print('train_data_idxs:')
        pprint(train_data_idxs[:10])
        print('Number of training data points: %d' % len(train_data_idxs))

        if model_name.lower() == "hyperplus":
            model = HypERPlus(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "hype":
            model = HypE(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "hyper":
            model = HypER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "distmult":
            model = DistMult(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "conve":
            model = ConvE(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif model_name.lower() == "complex":
            model = ComplEx(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        print('Model parameters:', [value.numel() for value in model.parameters()])

        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        print('sample ER:', er_vocab_pairs[0])
        print(er_vocab[er_vocab_pairs[0]])
        print(len(er_vocab_pairs))

        print('train_data_idxs:', train_data_idxs[:10])
        print("Starting training...")

        for it in range(1, self.num_iterations + 1):

            model.train()
            losses = []
            np.random.shuffle(train_data_idxs)

            for j in range(0, len(er_vocab_pairs), self.batch_size):
            # for j in range(1):
                data_batch, targets = self.get_batch(train_data_idxs, er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])
                e2_idx = torch.tensor(data_batch[:,2])
                ec_idx = torch.tensor(data_batch[:,3])

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    ec_idx = ec_idx.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx, ec_idx)

                # if self.label_smoothing:
                #     targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))

                # loss = model.loss(predictions)
                loss = model.loss(predictions)
                print('loss:', loss)
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
                self.evaluate(model, d.valid_data)
                # if not it % 2:
                #     print("Test:")
                #     self.evaluate(model, d.test_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="HypERPlus", nargs="?",
                    help='Which algorithm to use: HypERPlus, HypER, ConvE, DistMult, or ComplEx')
    # parser.add_argument('--dataset', type=str, default="FB15k-237", nargs="?",
    #                 help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    parser.add_argument('--dataset', type=str, default="WN18", nargs="?",
                    help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    args = parser.parse_args()

    model_name = args.algorithm
    dataset = args.dataset
    data_dir = "data/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(model_name, num_iterations=1, batch_size=128, learning_rate=0.001,
                            decay_rate=0.99, ent_vec_dim=200, rel_vec_dim=200, cuda=False,
                            input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2,
                            in_channels=1, out_channels=32, filt_h=1, filt_w=9, label_smoothing=0.1)
    experiment.train_and_eval()
