from collections import defaultdict
from sentence_data import *
import gzip
from parser_wrapper import ParserOutputWrapper as parser_w
import os
import operator
import time
from collections import namedtuple
from abc import ABCMeta, abstractproperty


class Pattern(object):
    FIRST_WILDCARD = 'X'
    SECOND_WILDCARD = 'Y'
    POS_TAG = 'NN'#JJ'
    POSSIBLE_TAGS = ['NN', 'NNS']

    def __init__(self,pattern):
        self.org_pattern = pattern
        self.formatted = pattern.replace(self.FIRST_WILDCARD,self.POS_TAG).replace(self.SECOND_WILDCARD,self.POS_TAG)


    # def is_match(self,string):


class PatternMatch(object):
    def __init__(self,string):
        self._string = string
        self._count = 0
        self._opposite_count = 0

    def total_count(self):
        return self._count + self._opposite_count


class PatternData(Pattern) :

    def __init__(self,pattern):
        super(PatternData,self).__init__(pattern)
        self._matching_strings = defaultdict(int)

    def add_matching_string(self,string):
        self._matching_strings[string] += 1

    @property
    def matching_strings_list(self):
        return self._matching_strings.keys()

    @property
    def matching_strings(self):
        return self._matching_strings.keys()

    @property
    def sorted_matching_strings(self):
        return sorted(self._matching_strings.items(),key=operator.itemgetter(1), reverse=True)


class PatternDataHandler(object):

    def __init__(self,initial_patterns_list, parser_output_file_name,read_file_format='rb'):
        self.parser_file = parser_output_file_name
        self.read_format = read_file_format
        self.pattern_to_data = {}#defaultdict(PatternData)

        patterns_data = [PatternData(pattern) for pattern in initial_patterns_list]
        self.pattern_to_data ={pattern.formatted:pattern for pattern in patterns_data}

        patterns_len = [len(pattern.split()) for pattern in self.pattern_to_data.keys()]
        self.min_pattern_len = min(patterns_len)
        self.max_pattern_len = max(patterns_len)

    def extract_patterns_matching(self):
        with gzip.open( self.parser_file, self.read_format) as f:
            sentence = []
            sentence_id = 0
            for line in f:
                if line != '\n':
                    data = line.split('\t')
                    word = data[parser_w.TOKEN_COLUMN].lower()
                    pos = data[parser_w.POS_COLUMN]
                    sentence.append((word,pos))
                else:
                    sentence_data = SentenceData(sentence)
                    self.extract_pattern_in_sentence(sentence_data)
                    sentence_id += 1
                    sentence = []
                    if (sentence_id % 1000000 == 0):
                        print "finished process sentence {}".format(sentence_id)
                        # break


    def extract_pattern_in_sentence(self,sentence):
        '''
        :param sentence:(SentenceData)
        :return:
        '''

        formatted_sentence = [Pattern.POS_TAG if word_data.pos in Pattern.POSSIBLE_TAGS
                              else word_data.word for word_data in sentence.words_data_sequence]
        for i in xrange(0,sentence.len-self.min_pattern_len+1):
            for j in xrange(i+self.min_pattern_len , i+self.max_pattern_len):
                string = " ".join(formatted_sentence[i:j])
                if self.pattern_to_data.has_key(string):
                    matched_string = " ".join(sentence.words_sequence[i:j])
                    self.pattern_to_data[string].add_matching_string(matched_string)

                    break


    def export_results(self,output_folder):

        import time

        localtime = time.asctime( time.localtime(time.time()) )
        output_folder = output_folder+'_'+localtime
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for (pattern,pattern_data) in self.pattern_to_data.iteritems():
            file_name = output_folder + '/' + pattern_data.org_pattern
            with open(file_name,'a') as f:
                rows = ['{}\t{}'.format(x[0],x[1]) for x in pattern_data.sorted_matching_strings]
                f.write("\n".join(rows))
