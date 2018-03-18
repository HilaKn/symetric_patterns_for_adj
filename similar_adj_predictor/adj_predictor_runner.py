from prediction_model import MLPNet, ShallowNet
from config_params import *
from torch.autograd import Variable
import torch
import numpy as np
import random
import operator
from scipy import spatial
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from data_handler import AdjNounAttribute
from word2vec_wrapper import we_model

attr_pred_res = "results/attr_pred_res"
adj_pred_res = "results/adj_pred_res"

class PredictorRunner(object):

    def __init__(self, data_handler, model_path, train_flag=True):
        self.__data_handler = data_handler
        self.cuda = CUDA_FLAG and torch.cuda.is_available()
        print "Using cuda: {}".format(self.cuda)
        self.nn_model = ShallowNet(D_IN, D_OUT)#MLPNet(D_IN, D_OUT, D_HIDDEN) #
        if  self.cuda:
            self.nn_model.cuda()
        self.criterion = torch.nn.MSELoss(size_average=True)#Mean Square Error loss
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LR)
        self.train_flag = train_flag
        self.model_path = model_path


    # def __prepare_data(self):
    #     pass
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr = LR * (0.1 ** (epoch // 20))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def __train_model(self):
        torch.set_num_threads(THREADS)
        for epoch in xrange(1, EPOCHS+1):
            running_loss = 0.0
            print "Epoch: {}".format(epoch)
            self.adjust_learning_rate(epoch)
            for batch_idx, (x_train, y_train, org_data) in enumerate(self.__data_handler.train_loader):
                # print org_data
                if self.cuda:
                    x_train, y_train = x_train.cuda(), y_train.cuda()
                else:
                    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
                x = Variable(x_train, requires_grad = True)
                y = Variable(y_train, requires_grad=False)

                if BATCH_SIZE == 1:
                    x.unsqueeze(0)
                    y.unsqueeze(0)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.nn_model(x)

                # Compute and print loss
                loss = self.criterion(y_pred, y)


                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if (batch_idx+1) % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x_train), len(self.__data_handler.train_loader.dataset),
                    100. * batch_idx / len(self.__data_handler.train_loader), running_loss/(len(x_train))))
                    running_loss = 0.0
                #
                # if (batch_idx + 1) % TEST_INTERVAL == 0:
                #     print "Batch {} - Test model so far".format(batch_idx)
                    # self.__intermediate_test()

        # self.__intermediate_test()

    # def __intermediate_test(self):
    #     predictions = []
    #     correct = 0.0
    #     iter = 1
    #     for idx,data_s in enumerate(self.__data_handler.train_ds.adj_noun_pairs):
    #         iter+=1
    #         if iter%TEST_INTERVAL == 0:
    #             print "Intermediate test {}".format(iter)
    #         # print org_data
    #         x = self.__data_handler.train_ds.train_data[idx]
    #         if self.cuda:
    #             x = x.cuda()
    #         else:
    #             x = torch.Tensor(x)
    #         data = Variable(x, volatile=True)
    #         output = self.nn_model(data)
    #         adj_vec = output.data.numpy().squeeze()
    #         # print adj_vec.shape
    #         unique_adj_mat = self.__data_handler.train_ds.adj_matrix
    #         # print unique_adj_mat.shape
    #
    #         sim =  np.dot( unique_adj_mat, adj_vec.T)
    #
    #
    #         # sim = cosine_similarity(adj_vec.reshape(1, -1), unique_adj_mat)#TODO: check shapes
    #         all_adj_idx = sim.argsort()[::-1]
    #         adj_all_preds = [self.__data_handler.train_ds.unique_adj_list[i] for i in list(all_adj_idx.squeeze())]
    #
    #         adj_preds = adj_all_preds[:EVAL_TOP_K_]
    #         # correct_pred_idx = attr_all_preds.index(org_data.attr)
    #         adj_pred_str = "|".join(adj_preds)
    #         predictions.append([data_s.adj, data_s.noun ,adj_pred_str])
    #         if adj_preds[0] == self.__data_handler.train_ds.adj_targets[idx]:
    #             correct +=1
    #         # if len(predictions) > 1999:
    #         #     print "stop intermediate test for 2000 items"
    #         #     break
    #
    #     print "-------intermediate test results------"
    #     print "correct = {}. total = {}. accuracy = {}".format(correct, len(predictions),
    #                                                            correct/len(predictions))
    #
    #
    #     output_file = "{}_{}".format(adj_pred_res, "intermediate_heiplas_dev")
    #     with open(output_file,'w') as file:
    #         for idx,pred in enumerate(predictions):
    #             string =  " ".join(pred)
    #             print >>file,string

    def __save_model(self):
        print "Start saving model to: {}".format(self.model_path)
        torch.save(self.nn_model.state_dict(), self.model_path)
        print "Done saving model to pickle file"

    def __load_model(self):
        print "Start loading model from: {}".format(self.model_path)
        self.nn_model.load_state_dict(torch.load(self.model_path))
        print "Done loading adj prediction model"

    def __test(self):
        print "Start testing"
        #run the model over HeiPLAS-dev adj-noun pairs
        #calc cosine similarity between the output to all the attributes
        #calc accuracy based on the best attribute
        test_loss = 0
        correct = 0.0
        correct_in_K = 0.0
        predictions = []
        for data, target, org_data in self.__data_handler.test_loader:
            # print org_data
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target =torch.Tensor(data), torch.Tensor(target)
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.nn_model(data)
            adj_vec = output.data.cpu().numpy().squeeze()
            # adj_rep_vec = (adj_vec + we_model[org_data[0]])/2.0
            # print "adj_rep_vec shape = {}".format(adj_rep_vec.shape)
            # sum_of_diff = np.sum(np.abs(adj_vec-adj_rep_vec))
            # cos_adj_sim = np.dot(adj_rep_vec,adj_vec)
            # print "diff to adj_vec = {}. cosine: {}".format(sum_of_diff, cos_adj_sim)
            # print "adj_vec shape = {}".format(adj_vec.shape)
            # adj_vec_reshaped = adj_vec.reshape(1,-1)
            # print "adj_vec shape after reshape = {}".format(adj_vec_reshaped.shape)
            unique_att_mat = self.__data_handler.test_ds.attr_matrix
            # print "unique_att_mat shape = {}".format(unique_att_mat.shape)
            sim = np.dot(unique_att_mat, adj_vec.T)#cosine_similarity(adj_vec_reshaped, unique_att_mat)#TODO: check shapes
            # print "sim shape= {}".format(sim.shape)
            all_attr_idx = sim.argsort()[::-1]
            # print "all_attr_idx shape = {}".format(all_attr_idx.shape)
            # print all_attr_idx
            attr_all_preds = [self.__data_handler.test_ds.unique_attributes[i] for i in list(all_attr_idx.squeeze())]
            # print "attr_all_preds len = {}".format(len(attr_all_preds))
            adj_preds = attr_all_preds[:EVAL_TOP_K_]
            # print adj_preds
            # correct_pred_idx = attr_all_preds.index(org_data.attr)
            predictions.append([org_data[2][0].upper(), org_data[0][0], org_data[1][0],adj_preds[0].upper()])
            # print org_data

            if adj_preds[0] == org_data[2][0]:#.attr:
                correct += 1
            if  org_data[2][0] in adj_preds:
                correct_in_K += 1

        print"----unsupervised results-----"
        print "correct = {}, total: {}, accuracy: {}".format(correct, len(self.__data_handler.test_loader), correct/len(self.__data_handler.test_loader))
        print"correct_in_{} = {}, total: {}, accuracy: {}".format(EVAL_TOP_K_, correct_in_K, len(self.__data_handler.test_loader), correct_in_K/len(self.__data_handler.test_loader))

        output_file = "{}_{}".format(attr_pred_res,"heiplas_dev")
        with open(output_file,'w') as file:
            for idx,pred in enumerate(predictions):
                # print pred[0]
                string = ' '.join(pred)
                print >>file,string


    def __intrinsic_evaluation(self):
        #run the model over HeiPLAS-dev adj-noun pairs
        #calc cosine similarity between the output to all possible adjectives (from the original ds)
        #output each adj-noun pair with list of K nearest neighbours (sorted descending)
        #manually evaluate the results
        print "start intrinsic evaluation"
        predictions = []
        for data, target, org_data in self.__data_handler.test_loader:
            # print org_data
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target =torch.Tensor(data), torch.Tensor(target)
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.nn_model(data)
            adj_vec = output.data.cpu().numpy().squeeze()
            # print adj_vec.shape
            unique_adj_mat = self.__data_handler.train_ds.adj_matrix
            # print unique_adj_mat.shape
            sim = np.dot(unique_adj_mat, adj_vec.T)#TODO: check shapes
            all_adj_idx = sim.argsort()[::-1]
            adj_all_preds = [self.__data_handler.train_ds.unique_adj_list[i] for i in list(all_adj_idx.squeeze())]

            adj_preds = adj_all_preds[:EVAL_TOP_K_]
            # correct_pred_idx = attr_all_preds.index(org_data.attr)
            adj_pred_str = "|".join(adj_preds)
            predictions.append([org_data[2][0].upper(),org_data[0][0], org_data[1][0],adj_pred_str])

        print "Done intrinsic evaluation. saving results to file"
        output_file = "{}_{}".format(adj_pred_res, "heiplas_dev")
        with open(output_file,'w') as file:
            for idx,pred in enumerate(predictions):
                string =  " ".join(pred)
                print >>file,string



    def run(self):
        # self.__prepare_data()
        if self.train_flag:
            self.__train_model()
            self.__save_model()
        else:
            self.__load_model()

        self.__intrinsic_evaluation()
        self.__test()



