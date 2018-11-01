# std
import logging

# 3rd party
import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')


class HypER(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):

        super(HypER, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(
            kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (1 - self.filt_h + 1) * \
            (d1 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(d2, fc1_length)

    def init(self):

        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):

        # Hpyer network
        r = self.R(r_idx)
        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels,
                   self.filt_h, self.filt_w)
        k = k.view(len(e1_idx) * self.in_channels *
                   self.out_channels, 1, self.filt_h, self.filt_w)

        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))
        x = self.bn0(e1)
        x = self.inp_drop(x)
        x = x.permute(1, 0, 2, 3)

        # convnet
        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 -
                   self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        # regularisation
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # dot product by e2
        x = torch.mm(x, self.E.weight.transpose(1, 0))

        # bias
        x += self.b.expand_as(x)

        # prediction
        pred = F.sigmoid(x)

        return pred


class HypERPlus(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):

        super().__init__()

        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(
            kwargs["feature_map_dropout"])

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)

        fc_length = (1 - self.filt_h + 1) * \
            (d1 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(d2, fc1_length)
        self.register_parameter('b', Parameter(torch.zeros(1)))
        self.fc2 = torch.nn.Linear(400, 1)

        self.loss = self.contrastive_max_margin_loss

    def contrastive_max_margin_loss(self, predictions):

        scalar = torch.FloatTensor([0])
        contrast = predictions[:, 0] + predictions[:, 1]
        loss = torch.max(1 - contrast, scalar.expand_as(contrast))
        cost = torch.sum(loss)

        return cost

    def accuracy(self, predictions, targets):

        correct_prediction = torch.eq(torch.argmax(predictions, 1), torch.max(targets, 1)[1])
        accuracy = torch.mean(correct_prediction.float())
        logging.debug(f'accuracy: {accuracy}')

        return accuracy

    def init(self):

        # initialise word and relational embeddings with random normel ~N(0, 1)
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx, ec_idx):

        logging.debug('Begninning forward prop...')
        # only compute based on e2 batch
        # change target from e2 to relationship

        # Hpyer network
        r = self.R(r_idx)
        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(len(e1_idx) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))
        x = self.bn0(e1)
        x = self.inp_drop(x)
        x = x.permute(1, 0, 2, 3)

        # convnet
        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 -
                   self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        # regularisation
        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # get everything
        e2 = self.E(e2_idx).view(-1, 1, 1, self.E.weight.size(1))
        x2 = self.bn0(e2)
        x2 = self.inp_drop(x2)
        x2 = x2.permute(1, 0, 2, 3)

        # convnet
        x2 = F.conv2d(x2, k, groups=e2.size(0))
        x2 = x2.view(e2.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e2.size(3) - self.filt_w + 1)
        x2 = x2.permute(0, 3, 4, 1, 2)
        x2 = torch.sum(x2, dim=3)
        x2 = x2.permute(0, 3, 1, 2).contiguous()

        # regularisation
        x2 = self.bn1(x2)
        x2 = self.feature_map_drop(x2)
        x2 = x2.view(e2.size(0), -1)
        x2 = self.fc(x2)
        x2 = self.hidden_drop(x2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        # dynamic target
        ec = self.E(ec_idx).view(-1, 1, 1, self.E.weight.size(1))
        x3 = self.bn0(ec)
        x3 = self.inp_drop(x3)
        x3 = x3.permute(1, 0, 2, 3)

        # convnet
        x3 = F.conv2d(x3, k, groups=ec.size(0))
        x3 = x3.view(ec.size(0), 1, self.out_channels, 1 - self.filt_h + 1, ec.size(3) - self.filt_w + 1)
        x3 = x3.permute(0, 3, 4, 1, 2)
        x3 = torch.sum(x3, dim=3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()

        # regularisation
        x3 = self.bn1(x3)
        x3 = self.feature_map_drop(x3)
        x3 = x3.view(ec.size(0), -1)
        x3 = self.fc(x3)
        x3 = self.hidden_drop(x3)
        x3 = self.bn2(x3)
        x3 = F.relu(x3)

        logging.debug(f'x size: {x.size()}')
        logging.debug(f'x2 size: {x2.size()}')

        x_in_e = torch.cat((x, x2), 1)
        x_in_c = torch.cat((x, x3), 1)

        logging.debug(f'x_in_e: {x_in_e}')
        logging.debug(f'x_in_c: {x_in_c.size()}')

        # fully-connected classification layer
        logits_e = self.fc2(x_in_e)
        logits_c = self.fc2(x_in_c)

        logging.debug(f'logits_e: {logits_e.size()}')
        logging.debug(f'logits_c: {logits_c.size()}')

        # # dot product by e2 and ec
        # logits_e = torch.mm(x, x2.transpose(1, 0))
        # logits_c = torch.mm(x, x3.transpose(1, 0))

        # logging.debug(f'logits_e size: {logits_e.size()}')
        # logging.debug(f'logits_c size: {logits_c.size()}')

        # bias
        logits_e = logits_e + self.b.expand_as(logits_e)
        logits_c = logits_c + self.b.expand_as(logits_c)

        logits = torch.cat((logits_e, logits_c), 1)

        logging.debug(f'logits size: {logits.size()}')

        # prediction
        pred = logits

        return pred


class ConvE(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):

        super(ConvE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(
            kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels,
                                     (self.filt_h, self.filt_w), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (20 - self.filt_h + 1) * \
            (20 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 10, 20)
        r = self.R(r_idx).view(-1, 1, 10, 20)
        x = torch.cat([e1, r], 2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred


class DistMult(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):
        super(DistMult, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()
        self.bn0 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        r = self.R(r_idx)
        e1 = self.bn0(e1)
        e1 = self.inp_drop(e1)
        pred = torch.mm(e1 * r, self.E.weight.transpose(1, 0))
        pred = F.sigmoid(pred)
        return pred


class ComplEx(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):
        super(ComplEx, self).__init__()
        self.Er = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.Rr = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.Ei = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.Ri = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.loss = torch.nn.BCELoss()
        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.Er.weight.data)
        xavier_normal_(self.Rr.weight.data)
        xavier_normal_(self.Ei.weight.data)
        xavier_normal_(self.Ri.weight.data)

    def forward(self, e1_idx, r_idx):
        e1r = self.Er(e1_idx)
        rr = self.Rr(r_idx)
        e1i = self.Ei(e1_idx)
        ri = self.Ri(r_idx)
        e1r = self.bn0(e1r)
        e1r = self.inp_drop(e1r)
        e1i = self.bn1(e1i)
        e1i = self.inp_drop(e1i)
        pred = torch.mm(e1r * rr, self.Er.weight.transpose(1, 0)) +\
            torch.mm(e1r * ri, self.Ei.weight.transpose(1, 0)) +\
            torch.mm(e1i * rr, self.Ei.weight.transpose(1, 0)) -\
            torch.mm(e1i * ri, self.Er.weight.transpose(1, 0))
        pred = F.sigmoid(pred)
        return pred


class HypE(torch.nn.Module):

    def __init__(self, d, d1, d2, **kwargs):

        super(HypE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        r_dim = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.R = torch.nn.Embedding(len(d.relations), r_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(
            kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (10 - self.filt_h + 1) * \
            (20 - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):

        e1 = self.E(e1_idx).view(-1, 1, 10, 20)

        r = self.R(r_idx)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = r.view(-1, self.in_channels, self.out_channels,
                   self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels *
                   self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 10 -
                   self.filt_h + 1, 20 - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred
