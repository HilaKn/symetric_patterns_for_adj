import argparse
from collections import namedtuple, defaultdict
from word2vec_wrapper import we_model, we_wrapper

AdjNoun = namedtuple("AdjNoun", 'adj, noun')
AdjCount = namedtuple("AdjCount", 'adj, count')

class DataItem(object):
    def __init__(self, adj, noun, adj_counts_list):
        self.adj = adj
        self.noun = noun
        self.target_adj_list = adj_counts_list
        self.total_count = sum([adj_count.count for adj_count in adj_counts_list])

class Triplet(object):

    def __init__(self,data_item, target_adj, pair_weight, triplet_weight):
        self.adj = data_item.adj
        self.noun = data_item.noun
        self.target_adj = target_adj.adj
        self.count = target_adj.count
        self.adj_noun_weight = pair_weight
        self.triple_weight = triplet_weight

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
    total_pairs = 0
    output_dataset = []
    iter = 0
    with open(args.input_file) as f:
        for row in f:
            iter+=1
            print "iter = {}".format(iter)
            row_data = row.rstrip("\r\n").split("\t")
            adj_count_list = row_data[1].split("|")
            adj,noun = row_data[0].split()
            # target_adj_list = []
            # for item in adj_count_list:
            #     item_data = item.split()
            #     if len(item_data) < 2:
            #         print "empty target adj = [{}]".format(item_data)
            #     else:
            #         adj = item_data[0].strip("()")
            #         count = int(item_data[1].strip("()"))
            #         adj_count = AdjCount(adj,count)
            #         target_adj_list.append(adj_count)
            target_adj_list = [AdjCount(item.split()[0].strip("()"),item.split()[1].strip("()")) for item in adj_count_list if len(item.split()) >1]
            filtered_tar_adj_list = [tar_adj for tar_adj in target_adj_list if we_wrapper.is_all_in_vocab([ adj, noun, tar_adj.adj])]
            total_for_pair = sum([int(adj_count.count) for adj_count in filtered_tar_adj_list])
            for tar_adj in filtered_tar_adj_list:
                weight = float(tar_adj.count)/total_for_pair
                out_raw_data = RawData(adj, noun, tar_adj.adj, weight)
                output_dataset.append(out_raw_data)
            if filtered_tar_adj_list:
                total_pairs += 1


    print "{} out of vocab unique words".format(len(we_wrapper.total_out_of_vocab))
    with open("out_of_vocab_words", "w")as f:
        f.write("\n".join(list(we_wrapper.total_out_of_vocab)))

    uniform_weight = 1.0/total_pairs
    map(lambda x: x.update_weight(x.weight*uniform_weight), output_dataset)

    with open(args.output_file, "w") as f:
        for item in output_dataset:
            f.write(str(item) +"\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data for adjective prediction task')
    parser.add_argument('input_file',help='file containing the initial data set with adj_noun pairs and matching adj-count list')
    parser.add_argument('output_file',help='output_file_name')
    args = parser.parse_args()
    run()
