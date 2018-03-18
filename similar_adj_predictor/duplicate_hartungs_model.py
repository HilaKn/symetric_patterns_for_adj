import gensim
from torch.autograd import Variable
import torch
import numpy as np
import random
import operator
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

BATCH_SIZE = 5
D_IN = 600 #The dimension of 2 concatenated word2vec for adjective and noun
D_OUT = 300 #word2vec dimension
EPHOCS = 15
word2vec_text_normed_path = "/home/h/data/word2vec/word2vec_text"
in_file = "../syntactic_dep_merged/post_analysis/triplets_only_top_prob_weights"

class Model(nn.Module):
    def __init__(self, D_in, D_out):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(D_in,D_out,bias=True)
        weights = np.concatenate((np.identity(D_out), np.identity(D_out)),axis=1) #initailize weight as in the paper
        self.linear_1.weight.data = torch.Tensor(weights)

    def forward(self, x):

        return self.linear_1(x)


class AdjNounAdj(object):
    def __init__(self,adj,noun,tar_adj):
        self.adj = adj
        self.noun = noun
        self.tar_adj = tar_adj

def read_data(file_path):
    with open(file_path) as f:
        input_list = [line.split() for line in f.readlines()]
    data = [AdjNounAdj(item[0],item[1],item[2]) for item in input_list]
    return data[0:800]


#########MAIM#############
def batch_training(batch_size = BATCH_SIZE, epochs = EPHOCS):
    running_loss = 0.0
    indices = range(y_train.shape[0])

    for epoch in range(epochs):
        print "Epoch: {}".format(epoch)
        random.shuffle(indices)
        for i in xrange(batch_size, y_train.shape[0] + batch_size, batch_size):
            if i >= y_train.shape[0]:
                current_indecies = indices[i - batch_size:y_train.shape[0] - 1]
            else:
                current_indecies = indices[i - batch_size:i]

            x = Variable(torch.Tensor(x_train[current_indecies]))
            y = Variable(torch.Tensor(y_train[current_indecies]), requires_grad=False)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = nn_model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print(epoch, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  #
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print "Finished batch training"

def online_training(epochs = EPHOCS ):
    running_loss = 0.0
    indices = range(y_train.shape[0])
    for epoch in range(epochs):
        print "Epoch: {}".format(epoch)
        random.shuffle(indices)
        for i in indices:

            x = Variable(torch.Tensor(x_train[[i]]))
            y = Variable(torch.Tensor(y_train[[i]]), requires_grad=False)

            # pytorch doesn't support directly in training without batching so this is kind of a hack
            x.unsqueeze(0)
            y.unsqueeze(0)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = nn_model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            print(epoch, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 1000 == 999:  #
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    print "Finished online training"


def test():

    weights = nn_model.linear_1.weight.data.numpy()
    bias = nn_model.linear_1.bias.data.numpy()
    print "before filter missing words, testing samples: " + str(len(test_triplets))
    #Filter test samples that their words are missing from the word2vec vocabulary
    filtered_test_samp = [samp for samp in test_triplets
                          if samp.adj in model.vocab and samp.noun in model.vocab and samp.tar_adj in model.vocab]
    print "after filter missing words, testing samples: " + str(len(filtered_test_samp))
    x_test = np.array([np.concatenate((model.word_vec(samp.adj), model.word_vec(samp.noun)))for samp in filtered_test_samp])
    y_test = np.array([model.word_vec(samp.tar_adj) for samp in filtered_test_samp])
    tar_vecs = {tar_adj: model.word_vec(tar_adj) for tar_adj in tar_adjectives if tar_adj in model.vocab}

    print "x test shape: " + str(x_test.shape)
    print "y_test: " + str(y_test.shape)
    print "weights shape: {}".format(weights.shape)

    x_test_matrix = (np.dot(weights, np.transpose(x_test)).T+bias.T).T
    print "x_test matrix shape = {}".format(x_test_matrix.shape)

    # check P@1 and P@5 accuracy
    correct = 0.0
    top_5_correct = 0.0
    predictions = []
    for i in xrange(0, x_test_matrix.shape[1]):
        y_pred = x_test_matrix[:, [i]]
        #calculate cosine similarity for normalized vectors
        cosine_sims = {adj: np.dot(y_pred.T, tar_vecs[adj]) for adj in tar_vecs.keys()}
        sorted_sims = dict(sorted(cosine_sims.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
        most_sim_adj = max(sorted_sims, key=lambda i: sorted_sims[i])
        predictions.append(most_sim_adj)
        if most_sim_adj == filtered_test_samp[i].tar_adj:
            correct += 1
        if filtered_test_samp[i].tar_adj in sorted_sims.keys():
            top_5_correct += 1
    print "correct: {} from total: {}. Accuracy: {}".format(correct, y_test.shape[0], correct / y_test.shape[0])
    print "top 5 correct: {} from total: {}. Accuracy: {}".format(top_5_correct, y_test.shape[0],
                                                                  top_5_correct / y_test.shape[0])

    output_file = "results/hartungs_for_adj_tar"
    with open(output_file,'w') as file:
        for idx,pred in enumerate(predictions):
            samp = filtered_test_samp[idx]
            string =  " ".join([samp.adj,samp.noun,samp.tar_adj])
            print >>file,string



if __name__ == '__main__':

    dev_triplets = read_data(in_file)
    tar_adjectives = {triplet.tar_adj for triplet in dev_triplets}
    test_triplets= dev_triplets#read_HeiPLAS_data(test_file_path)

    # load pre-trained, normalized word2ec
    model = gensim.models.KeyedVectors.load(word2vec_text_normed_path, mmap='r')  # mmap the large matrix as read-only
    model.syn0norm = model.syn0


    #generate trainig vectors
    x_train = np.array([np.concatenate((model.word_vec(samp.adj),model.word_vec(samp.noun)))for samp in dev_triplets])
    y_train = np.array([model.word_vec(samp.tar_adj) for samp in dev_triplets])

    print "x shape: "+ str(x_train.shape)
    print "y_train: " + str(y_train.shape)

    #prepare NN model
    nn_model = Model(D_IN, D_OUT)
    criterion = torch.nn.MSELoss(size_average=True)#Mean Squa,re Error loss
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4)

    # batch_training(BATCH_SIZE)

    online_training()

    test()

    print "Done!!!!!"