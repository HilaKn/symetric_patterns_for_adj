from os import listdir
from os.path import join
import os
from patterns_code.patterns_data import Pattern
import operator
from collections import namedtuple, defaultdict

AdjNoun = namedtuple("AdjNoun", 'adj, noun')
AdjCount = namedtuple("AdjCount", 'adj, count')

class AdjCountCollection(object):

    def __init__(self):
        self.__adj_to_count = defaultdict(int)

    def add(self, adj, count):
        self.__adj_to_count[adj] += count

    @property
    def sorted_data(self):
        return sorted(self.__adj_to_count.items(),key=operator.itemgetter(1), reverse=True)


    def get_keys(self):
        return [key for key,value in self.sorted_data]

    def __str__(self):
        out_list = ["({} {})".format(adj,count) for adj,count in self.sorted_data]
        return "|".join(out_list)

class TripletHandler(object):

    def __init__(self, input_folder, output_file):
        self.input_files = [join(input_folder, f) for f in listdir(input_folder) if os.path.isfile(join(input_folder, f))]
        self.output_path = os.path.join(input_folder, output_file)
        self.output_2_path = os.path.join(input_folder, "clean_data")
        self.adj_noun_to_adj_list = defaultdict(AdjCountCollection)


    def load_data(self):
        for file in self.input_files:
            pattern = Pattern(os.path.basename(file))
            adj1_idx = pattern.wildcards_idx[0]
            adj2_idx = pattern.wildcards_idx[1]
            with open(file) as f:
                for line in f:
                    data_row = line.rstrip('\n').split('\t')
                    pattern_str = data_row[0].split()
                    count = int(data_row[1])
                    adj_1 = pattern_str[adj1_idx]
                    adj_2 = pattern_str[adj2_idx]
                    noun = pattern_str[-1]

                    self.adj_noun_to_adj_list[AdjNoun(adj_1, noun)].add(adj_2, count)
                    self.adj_noun_to_adj_list[AdjNoun(adj_2, noun)].add(adj_1, count)

    def generate_output(self):
        with open(self.output_path, 'w') as f:
            for adj_noun, adj_count_coll in self.adj_noun_to_adj_list.iteritems():
                key_str = " ".join([adj_noun.adj, adj_noun.noun])
                value_str = str(adj_count_coll)
                output = "{}\t{}\n".format(key_str, value_str)
                f.write(output)

        with open(self.output_2_path, 'w') as f:
            for adj_noun, adj_count_coll in self.adj_noun_to_adj_list.iteritems():
                key_str = " ".join([adj_noun.adj, adj_noun.noun])
                value_str =",".join( [key for key in  adj_count_coll.get_keys()])
                output = "{}\t{}\n".format(key_str, adj_count_coll.get_keys())
                f.write(output)


    def run(self):
        self.load_data()
        self.generate_output()


