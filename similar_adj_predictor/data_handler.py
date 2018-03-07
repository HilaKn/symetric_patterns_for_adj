import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple, defaultdict
from word2vec_wrapper import we_model
import torch
from config_params import *
import numpy as np
from collections import namedtuple

AdjNounAttribute = namedtuple("AdjNounAttribute", 'adj, noun, attr')

class HeiPlasDs(data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.dev_data = []
        self.dev_targets = []
        self.dev_adj_noun_attr_list = []
        self.unique_attributes = []
        self.__unique_attr_matrix = np.array

        with open(self.input_file) as f:
            attr_set = set()
            for row in f:
                row_data = row.rstrip("\n").split("\t")
                adj_vec = we_model[row_data[0]]
                noun_vec = we_model[row_data[1]]
                attr = row_data[2].lower()
                attr_set.add(attr)
                self.dev_adj_noun_attr_list.append(AdjNounAttribute(row_data))
                in_dev = np.concatenate((adj_vec,noun_vec))
                self.dev_data.append(in_dev)
                self.dev_targets.append(attr)

        self.unique_attributes = list(attr_set)

    @property
    def attr_matrix(self):
        if not self.__unique_attr_matrix:
            self.__unique_attr_matrix = np.array([we_model[attr] for attr in self.unique_attributes]).squeeze()
        return self.__unique_attr_matrix

    def __getitem__(self, index):
        attr = self.dev_targets[index]
        input, target, org_data = self.dev_data[index], we_model[attr], self.dev_adj_noun_attr_list[index]
        return input, target, org_data

    def __len__(self):
        return len(self.dev_data)


class ANADataset(data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.train_data = []
        self.samp_weights = []
        self.train_targets = []
        self.unique_adj_list = []
        self.__unique_adj_matrix = []

        unique_adj = set()
        with open(self.input_file) as f:
            for row in f:
                row_data = row.split("\t")
                adj_vec = we_model[row_data[0]]
                unique_adj.add(row_data[0])
                noun_vec = we_model[row_data[1]]
                out_train = we_model[row_data[2]]
                in_train = np.concatenate((adj_vec,noun_vec))
                self.train_data.append(in_train)
                self.train_targets.append(out_train)
                self.samp_weights.append(float(row_data[3]))

        self.unique_adj_list = list(unique_adj)

    @property
    def adj_matrix(self):
        if not self.__unique_adj_matrix:
            self.__unique_adj_matrix = np.array([we_model[adj] for adj in self.unique_adj_list]).squeeze()
        return self.__unique_adj_matrix

    def __getitem__(self, index):
        input, target = self.train_data[index], self.train_targets[index]
        return input, target

    def __len__(self):
        return len(self.train_data)

class DataHandler(object):

    def __init__(self, train_data_file, test_data_file):
        self.train_ds = ANADataset(train_data_file)
        self.train_sampler = WeightedRandomSampler(weights = self.train_ds.weights, num_samples = len(self.train_ds)*SAMPLE_FACTOR, replacement=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_ds,
                                           sampler = self.train_sampler,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=DATA_LOADER_WORKERS)

        self.test_ds = HeiPlasDs(test_data_file)
        self.test_sampler = SequentialSampler(self.test_ds)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_ds,
                                                       sampler=self.test_sampler,
                                                       batch_size=TEST_BATCH_SIZE,
                                                       shuffle=False)





    @property
    def train_size(self):
        return len(self.train_ds)

    @property
    def test_size(self):
        return len(self.test_ds)






