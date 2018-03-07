from abc import ABCMeta, abstractproperty
import gzip
from parser_wrapper import ParserOutputWrapper as parser_w
from sentence_data import SentenceData
import os
from parser_wrapper import ParserOutputWrapper as pw


class DataTypes:
    WIKIPEDIA = "wiki"
    GOOGLE_N_GRAMS = "google"
    SYNTACTIC_N_GRAMS = "syntactic"


class DataWrapper(object):

    __metaclass__ = ABCMeta

    def __init__(self, input_path):
        self.input_path = input_path

    @abstractproperty
    def data_collection(self):
        pass


class ParsedWikiWrapper(DataWrapper):

    def __init__(self, input_file):
        super(ParsedWikiWrapper,self).__init__(input_file)


    def data_collection(self):
        with gzip.open( self.input_path, 'rb') as f:
            sentence = []

            for line in f:
                if line != '\n':
                    data = line.split('\t')
                    word = data[parser_w.TOKEN_COLUMN].lower()
                    pos = data[parser_w.POS_COLUMN]
                    head = int(data[parser_w.DEP_ID_COLUMN])-1
                    sentence.append((word,pos,head))
                else:
                    sentence_data = SentenceData(sentence)
                    yield sentence_data


class SyntacticNgrams(DataWrapper):

    FILES_PREFIX = "extended-triarcs"

    def __init__(self, input_folder):
        super(SyntacticNgrams,self).__init__(input_folder)
        files = [file for file in  os.listdir(self.input_path) if file.endswith(".gz") and file.startswith(self.FILES_PREFIX)]
        self.ngrams_files = [os.path.join(self.input_path, file) for file in  files]
        print "Total {} files in {}".format(len(self.ngrams_files), self.FILES_PREFIX)


    def get_sentence_data(self, word_data):
        '''
        cease/VB/ccomp/0
        '''
        data = word_data.split('/')
        data_size = len(data)
        word = data[0].lower()
        pos = data[1]
        head = int(data[data_size-1])-1

        return (word,pos,head)

    def data_collection(self):
        file_number = 0
        for file in self.ngrams_files:
            print "File {} - start iterating over file = {}".format(file_number,os.path.basename(file))
            file_number += 1
            with gzip.open(file, 'rb') as f:
                for line in f:
                    try:
                        line_data = line.split("\t")
                        syntactic_data = line_data[1].split()#this is where the parsing data
                        syntactic_count = int(line_data[2])#total count of this syntactic ngram
                        sentence = [self.get_sentence_data(word_data) for word_data in syntactic_data]
                        if len([word_data for word_data in sentence if word_data[1] == pw.ADJ_TAG] ) >1:
                            sentence_data = SentenceData(sentence, syntactic_count)
                            yield sentence_data
                    except Exception as e:
                        print "error while iterating line: {}".format(line)
                        print "Error: {}".format(e)


class GoogleNgrams(DataWrapper):

    def __init__(self, input_folder):
        super(GoogleNgrams,self).__init__(input_folder)
        files = [file for file in  os.listdir(self.input_path) if file.endswith(".gz")]
        self.ngrams_files = [os.path.join(self.input_path, file) for file in  files]
        print "Total {} files in {}".format(len(self.ngrams_files), input_folder)

    def convert_pos_tag(self, pos_tag):
        if pos_tag == "ADJ":
            return pw.ADJ_TAG
        else:
            return pos_tag

    def get_sentence_data(self, word_data):

        data = word_data.split('_')
        word = data[0].lower()
        pos = self.convert_pos_tag(data[1])
        head = "<NO_NOUN>"

        return (word,pos,head)

    def data_collection(self):
        file_number = 1
        for file in self.ngrams_files:
            print "File {} - start iterating over file = {}".format(file_number,os.path.basename(file))
            self.get_data_from_single_file(file)


    def get_data_from_single_file(self,file):
        with gzip.open(file, 'rb') as f:
            for line in f:
                try:
                    line_data = line.split("\t")
                    words_data = line_data[0].split()#this is where mgram and pos are
                    count = int(line_data[2])#total count
                    sentence = [self.get_sentence_data(word_data) for word_data in words_data]
                    if len([word_data for word_data in sentence if word_data[1] == pw.ADJ_TAG] ) >1:
                        sentence_data = SentenceData(sentence, count)
                        yield sentence_data
                except Exception as e:
                    pass
                    # print "error while iterating line: {}".format(line)
                    # print "Error: {}".format(e)



def data_wrapper_factory(input_path, data_type):
    data_wrapper = None
    if data_type == DataTypes.WIKIPEDIA:
        data_wrapper = ParsedWikiWrapper(input_path)
    elif data_type == DataTypes.SYNTACTIC_N_GRAMS:
        data_wrapper = SyntacticNgrams(input_path)
    elif data_type == DataTypes.GOOGLE_N_GRAMS:
        data_wrapper = GoogleNgrams(input_path)
    return data_wrapper

