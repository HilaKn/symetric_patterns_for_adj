import argparse
from collections import namedtuple, defaultdict
from word2vec_wrapper import we_model, we_wrapper

class RawData(object):
    def __init__(self, adj, noun, target_adj, weight = 1.0):
        self.adj = adj
        self.noun = noun
        self.target_adj = target_adj
        self.weight = weight

    def update_weight(self, w):
        self.weight  = w

    def __str__(self):
        return "\t".join([self.adj, self.noun, self.target_adj, str(self.weight)])


def run():
    output_dataset = []
    iter = 0
    with open(args.input_file) as f:
        for row in f:
            iter+=1
            print "iter = {}".format(iter)
            row_data = row.rstrip("\r\n").split("\t")
            adj_count_list = row_data[1].split("|")
            adj,noun = row_data[0].split()
            for adj_count in adj_count_list:
                adj_count_data = adj_count.split()
                if len(adj_count_data) >1:
                    item = RawData(adj, noun, adj_count_data[0].strip("()"), int(adj_count_data[1].strip("()")))
                    if we_wrapper.is_all_in_vocab([item.adj, item.noun, item.target_adj]):
                        output_dataset.append(item)
                else:
                    print "adj_count <2: {}".format(adj_count_data)

    total_count = float(sum(item.weight for item in output_dataset))
    map(lambda x: x.update_weight(x.weight/total_count), output_dataset)


    with open(args.output_file, "w") as f:
        for item in output_dataset:
            f.write(str(item) +"\n")

def prepare_single_sample_per_adj_noun():
    output_dataset = []
    iter = 0
    with open(args.input_file) as f:
        for row in f:
            iter+=1
            print "iter = {}".format(iter)
            row_data = row.rstrip("\r\n").split("\t")
            adj_count_list = row_data[1].split("|")
            adj,noun = row_data[0].split()

            tar_adj_count = adj_count_list[0].strip("()").split()
            tar_adj, count = tar_adj_count[0], int(tar_adj_count[1])
            item = RawData(adj, noun, tar_adj, count)
            if we_wrapper.is_all_in_vocab([item.adj, item.noun, item.target_adj]):
                output_dataset.append(item)


    total_count = float(sum(item.weight for item in output_dataset))
    map(lambda x: x.update_weight(x.weight/total_count), output_dataset)


    with open(args.output_file, "w") as f:
        for item in output_dataset:
            f.write(str(item) +"\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data for adjective prediction task')
    parser.add_argument('input_file',help='file containing the initial data set with adj_noun pairs and matching adj-count list')
    parser.add_argument('output_file',help='output_file_name')
    args = parser.parse_args()
    # run()
    prepare_single_sample_per_adj_noun()