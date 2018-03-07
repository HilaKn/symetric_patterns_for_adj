from prediction_model import MLPNet
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

attr_pred_res = "results/attr_pred_res"
adj_pred_res = "results/adj_pred_res"

class PredictorRunner(object):

    def __init__(self, data_handler, model_path, train_flag=True):
        self.__data_handler = data_handler
        self.nn_model = MLPNet(D_IN, D_OUT, D_HIDDEN)
        self.criterion = torch.nn.MSELoss(size_average=True)#Mean Square Error loss
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LR)
        self.train_flag = train_flag
        self.model_path = model_path

    # def __prepare_data(self):
    #     pass

    def __train_model(self):
        running_loss = 0.0
        for epoch in xrange(1, EPOCHS+1):
            print "Epoch: {}".format(epoch)
            for batch_idx, (x_train, y_train) in enumerate(self.__data_handler.train_loader):
                x = Variable(torch.Tensor(x_train), requires_grad = False)
                y = Variable(torch.Tensor(y_train), requires_grad=False)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.nn_model(x)

                # Compute and print loss
                loss = self.criterion(y_pred, y)
                print(epoch, loss.data[0])

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if batch_idx % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x_train), len(self.__data_handler.train_loader.dataset),
                    100. * batch_idx / len(self.__data_handler.train_loader), loss.data[0]))


    def __save_model(self):
        torch.save(self.nn_model.state_dict(), self.model_path)

    def __load_model(self):
        self.nn_mode.load_state_dict(torch.load(self.model_path))

    def __test(self):
        #run the model over HeiPLAS-dev adj-noun pairs
        #calc cosine similarity between the output to all the attributes
        #calc accuracy based on the best attribute
        test_loss = 0
        correct = 0.0
        correct_in_K = 0.0
        predictions = []
        for data, target, org_data in self.__data_handler.test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.nn_model(data)
            adj_vec = output.data.numpy()

            sim = cosine_similarity(adj_vec, self.__data_handler.test_ds.unique_attr_matrix)#TODO: check shapes
            all_attr_idx = sim.argsort()[::-1]
            attr_all_preds = [self.__data_handler.test_ds.unique_attributes[i] for i in all_attr_idx]

            adj_preds = attr_all_preds[:EVAL_TOP_K_]
            # correct_pred_idx = attr_all_preds.index(org_data.attr)
            predictions.append(adj_preds[0])
            if adj_preds[0] == org_data.attr:
                correct += 1
            if org_data.attr in adj_preds:
                correct_in_K += 1

        output_file = "{}_{}".format(attr_pred_res, self.__data_handler.test_ds.input_file)
        with open(output_file,'w') as file:
            for idx,attr in enumerate(predictions):
                string = ' '.join(" ".join(org_data),attr.upper())
                print >>file,string

        print"----unsupervised results-----"
        print "correct = {}, total: {}, accuracy: {}".format(correct, len(self.__data_handler.test_loader), correct/len(self.__data_handler.test_loader))
        print"correct_in_{} = {}, total: {}, accuracy: {}".format(EVAL_TOP_K_, correct_in_K, len(self.__data_handler.test_loader), correct_in_K/len(self.__data_handler.test_loader))

    def __intrinsic_evaluation(self):
        #run the model over HeiPLAS-dev adj-noun pairs
        #calc cosine similarity between the output to all possible adjectives (from the original ds)
        #output each adj-noun pair with list of K nearest neighbours (sorted descending)
        #manually evaluate the results

        predictions = []
        for data, target, org_data in self.__data_handler.test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.nn_model(data)
            adj_vec = output.data.numpy()

            sim = cosine_similarity(adj_vec, self.__data_handler.unique_adj_matrix)#TODO: check shapes
            all_adj_idx = sim.argsort()[::-1]
            adj_all_preds = [self.__data_handler.unique_adj_matrix[i] for i in all_adj_idx]

            adj_preds = adj_all_preds[:EVAL_TOP_K_]
            # correct_pred_idx = attr_all_preds.index(org_data.attr)
            predictions.append(adj_preds)


        output_file = "{}_{}".format(adj_pred_res, self.__data_handler.test_ds.input_file)
        with open(output_file,'w') as file:
            for idx,adj_list in enumerate(predictions):
                string = ' '.join(" ".join(org_data),"|".join(adj_list))
                print >>file,string



    def run(self):
        # self.__prepare_data()
        if self.train_flag:
            self.__train_model()
            self.__save_model()
        else:
            self.__load_model()

        self.__test()
        self.__intrinsic_evaluation()




# class SupervisedModel(object):
#
#     def __init__(self, data_handler):
#         self.data = data_handler
#
#         nn_model = Model(D_IN, D_OUT)
#         self.nn_model = nn_model
#         self.criterion = torch.nn.MSELoss(size_average=True)#Mean Square Error loss
#         self.optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4)
#
#     def run(self):
#         self.online_training()
#         self.test()
#
#
#     def online_training(self,epochs = EPHOCS ):
#         running_loss = 0.0
#         y_train = self.data.y_train
#         x_train = self.data.x_train
#         indices = range(y_train.shape[0])
#         for epoch in range(epochs):
#             logger.info("Epoch: {}".format(epoch))
#             random.shuffle(indices)
#             for i in indices:
#
#                 x = Variable(torch.Tensor(x_train[[i]]))
#                 y = Variable(torch.Tensor(y_train[[i]]), requires_grad=False)
#
#                 # pytorch doesn't support directly in training without batching so this is kind of a hack
#                 x.unsqueeze(0)
#                 y.unsqueeze(0)
#
#                 # Forward pass: Compute predicted y by passing x to the model
#                 y_pred = self.nn_model(x)
#
#                 # Compute and print loss
#                 loss = self.criterion(y_pred, y)
#                 # print(epoch, loss.data[0])
#
#                 # Zero gradients, perform a backward pass, and update the weights.
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # print statistics
#                 # running_loss += loss.data[0]
#                 # if i % 100 == 99:  #
#                 #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#                 #     running_loss = 0.0
#         logger.info("Done online training")
#
#
#     def test(self):
#         we_wrapper = self.data.we_wrapper
#         weights = self.nn_model.linear_1.weight.data.numpy()
#
#         x_test =  self.data.x_test
#         y_test = self.data.y_test
#         attr_vecs = self.data.attr_vecs
#
#         print "attr_vecs size = {}".format(len(attr_vecs))
#         print "x test shape: " + str(x_test.shape)
#         print "y_test: " + str(y_test.shape)
#         print "weights shape: {}".format(weights.shape)
#
#         x_test_matrix = np.dot(weights, np.transpose(x_test))
#         print "x_test matrix shape = {}".format(x_test_matrix.shape)
#
#         # check P@1 and P@5 accuracy
#         correct = 0.0
#         top_5_correct = 0.0
#         correct_pred =[]
#         false_pred = []
#         results = []
#         for i in xrange(0, x_test_matrix.shape[1]):
#             y_pred = x_test_matrix[:, [i]]
#
#             #calculate cosine similarity for normalized vectors
#             cosine_sims = {attr: np.dot(y_pred.T, attr_vecs[attr]) for attr in attr_vecs.keys()}
#             sorted_sims = dict(sorted(cosine_sims.iteritems(), key=operator.itemgetter(1), reverse=True)[:K])
#             most_sim_attr = max(sorted_sims, key=lambda i: sorted_sims[i])
#             if most_sim_attr == self.data.test[i].attr:
#                 correct += 1
#                 correct_pred.append(self.data.test[i])
#             else:
#                 false_pred.append((self.data.test[i],most_sim_attr))
#             if self.data.test[i].attr in sorted_sims.keys():
#                 top_5_correct += 1
#             results.append((self.data.test[i],most_sim_attr))
#         logger.info("supervised results")
#         logger.info("correct: {} from total: {}. Accuracy: {}".format(correct, y_test.shape[0], correct / y_test.shape[0]))
#         logger.info("top 5 correct: {} from total: {}. Accuracy: {}".format(top_5_correct, y_test.shape[0],
#                                                                       top_5_correct / y_test.shape[0]))
#
#         with open(correct_predictions_file,'w') as file:
#             for item in correct_pred:
#                 # output = ' '.join([str(item), item[1].upper()])
#                 print >>file,item
#
#         with open(false_prediction_file,'w') as file:
#             for item in false_pred:
#                 output = ' '.join([str(item[0]), item[1].upper()])
#                 print >>file,output
#
#         with open(test_results,'w')as file:
#             for item in results:
#                 # output =  ' '.join([item[1].upper(), item[0]].adj, item[0].noun)
#                 print >>file,str(item[0])
