# std
import sys
import pickle
import logging
import argparse

# 3rd party
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_

# internal
from load_data import Data
from experiment import Experiment, ExperimentHypERPlus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="HypERPlus", nargs="?",
                        help='Which algorithm to use: ProxE, HypER, HypER, ConvE, DistMult, or ComplEx')
    parser.add_argument('--dataset', type=str, default="WN18", nargs="?",
                        help='Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR')
    args=parser.parse_args()

    model_name=args.algorithm
    dataset=args.dataset
    data_dir="data/%s/" % dataset

    torch.backends.cudnn.deterministic=True
    seed=42
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    d=Data(data_dir=data_dir, reverse=True)
    experiment=ExperimentHypERPlus(model_name, d, num_epoch=100, batch_size=128, learning_rate=0.001,
                            decay_rate=0.99, ent_vec_dim=200, rel_vec_dim=200, cuda=False,
                            input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2,
                            in_channels=1, out_channels=32, filt_h=1, filt_w=9, label_smoothing=0.1)
    experiment.train_and_eval()
