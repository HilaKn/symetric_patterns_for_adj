import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler, RandomSampler
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple, defaultdict
from word2vec_wrapper import we_model
import torch
from config_params import *
import numpy as np
from collections import namedtuple

AdjNounAttribute = namedtuple("AdjNounAttribute", 'adj, noun, attr')
AdjNoun = namedtuple("AdjNoun", 'adj, noun')

class HeiPlasDs(data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.dev_data = []
        self.dev_targets = []
        self.dev_adj_noun_attr_list = []
        self.unique_attributes = []
        self.__unique_attr_matrix = []

        with open(self.input_file) as f:
            attr_set = set()
            for row in f:
                attr,adj,noun = row.rstrip("\n").split()
                attr = attr.lower()
                attr_set.add(attr)
                adj_vec = we_model[adj]
                noun_vec = we_model[noun]

                self.dev_adj_noun_attr_list.append(AdjNounAttribute(adj,noun,attr))
                in_dev = np.concatenate((adj_vec,noun_vec))
                self.dev_data.append(in_dev)
                self.dev_targets.append(attr)

        attr_set.remove("good")
        self.unique_attributes = list(attr_set)

    @property
    def attr_matrix(self):
        # if not self.__unique_attr_matrix:
        #     self.__unique_attr_matrix = np.array([we_model[attr] for attr in self.unique_attributes]).squeeze()
        # return self.__unique_attr_matrix
        return np.array([we_model[attr] for attr in self.unique_attributes]).squeeze()

    def __getitem__(self, index):
        attr = self.dev_targets[index]
        input, target, org_data = np.array(self.dev_data[index]), np.array(we_model[attr]), self.dev_adj_noun_attr_list[index]
        return input, target, org_data

    def __len__(self):
        return len(self.dev_data)

# class AdjNou

class ANADataset(data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        self.train_data = []
        self.samp_weights = []
        self.train_targets = []
        self.unique_adj_list = []
        self.adj_noun_pairs = []
        self.adj_targets = []
        # self.__unique_adj_matrix = []

        unique_adj = set()
        counter = 0
        with open(self.input_file) as f:
            for row in f:
                counter+=1
                if counter % 10000 == 0:
                    print "processing {} row for ANADataset".format(counter)
                row_data = row.split("\t")
                adj_vec = we_model[row_data[0]]
                unique_adj.add(row_data[2])
                noun_vec = we_model[row_data[1]]
                out_train = we_model[row_data[2]]
                self.adj_targets.append(row_data[2])
                in_train = np.concatenate((adj_vec,noun_vec))
                self.train_data.append(in_train)
                self.train_targets.append(out_train)
                # self.samp_weights.append(1.0)
                self.samp_weights.append(float(row_data[3])) #TODO: get back to this
                self.adj_noun_pairs.append(AdjNoun(row_data[0], row_data[1]))
                # if counter > 80000:
                #     print "stop loading data at iteration: {}".format(counter)
                #     break

        self.unique_adj_list = list(unique_adj)

    @property
    def adj_matrix(self):
        # if not self.__unique_adj_matrix:
        #     self.__unique_adj_matrix = np.array([we_model[adj] for adj in self.unique_adj_list]).squeeze()
        # return self.__unique_adj_matrix
        return np.array([we_model[adj] for adj in self.unique_adj_list]).squeeze()

    def __getitem__(self, index):
        input, target, org_data = np.array(self.train_data[index]), np.array(self.train_targets[index]), self.adj_noun_pairs[index]
        return input, target, org_data

    def __len__(self):
        return len(self.train_data)

class DataHandler(object):

    def __init__(self, train_data_file, test_data_file):
        self.train_ds = ANADataset(train_data_file)
        self.train_sampler = WeightedRandomSampler(weights = self.train_ds.samp_weights, num_samples = len(self.train_ds)*SAMPLE_FACTOR, replacement=True)
        # self.rand_sampler = RandomSampler(self.train_ds)
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






