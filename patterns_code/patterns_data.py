from multiprocess import Process, Value, Pool, cpu_count, Lock, managers, Manager
from collections import defaultdict
from sentence_data import *
import gzip
from parser_wrapper import ParserOutputWrapper as parser_w
import os
import operator
import time
from collections import namedtuple
from abc import ABCMeta, abstractproperty, abstractmethod
from data_wrappers import *

NO_HEAD_NOUN = "<NO_NOUN>"
WORKERS = cpu_count() - 2
sent_locker = Lock()
lock = Lock()
sentence_counter = Value("i",0)

class Pattern(object):
    FIRST_WILDCARD = 'X'
    SECOND_WILDCARD = 'Y'
    POS_TAG = 'JJ'
    POSSIBLE_TAGS = ['JJ']
    NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']

    def __init__(self,pattern):
        self.org_pattern = pattern
        self.formatted = pattern.replace(self.FIRST_WILDCARD,self.POS_TAG).replace(self.SECOND_WILDCARD,self.POS_TAG)
        self.pattern_words = pattern.rstrip('\n').split()
        self.wildcards_idx = [idx for idx,word in enumerate(self.pattern_words) if word in [self.FIRST_WILDCARD,self.SECOND_WILDCARD]]



class PatternData(Pattern) :

    def __init__(self,pattern):
        super(PatternData,self).__init__(pattern)
        self._matching_strings = defaultdict(int)

    def add_matching_string(self,string, count):
        self._matching_strings[string] += count

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

    __metaclass__ = ABCMeta

    def __init__(self,initial_patterns_list, input_text_file,data_source_type,read_file_format='rb'):
        self.read_format = read_file_format
        self.pattern_to_data = {}#defaultdict(PatternData)

        patterns_data = [PatternData(pattern) for pattern in initial_patterns_list]
        self.pattern_to_data = Manager().dict({pattern.formatted:pattern for pattern in patterns_data})

        patterns_len = [len(pattern.split()) for pattern in self.pattern_to_data.keys()]
        self.min_pattern_len = min(patterns_len)
        self.max_pattern_len = max(patterns_len)
        self.data_wrapper = data_wrapper_factory(input_text_file, data_source_type)




    def extract_patterns_matching(self):
        sentence_id = 0
        for sentence_data in self.data_wrapper.data_collection():
            self.extract_pattern_in_sentence(sentence_data)
            sentence_id += 1
            # print "sentence {}".format(sentence_id)
            if sentence_id % 10000 == 0:
                        print "finished process sentence {}".format(sentence_id)
                        # break


    def extract_patterns_from_file(self, file):

        for sentence_data in self.data_wrapper.get_data_from_single_file(file):
            self.extract_pattern_in_sentence(sentence_data)
            with sent_locker:
                sentence_counter.value += 1
                counter = sentence_counter.value
                # print "p_id = {} sentence {}".format(os.getpid(),counter)
            if counter % 100000 == 0:
                        print "finished process sentence {}".format(counter)
                        # break


    def extract_patterns_matching_async(self):
        startTime = time.time()
        print "running on {} processors".format(WORKERS)
        pool = Pool(processes=WORKERS,initargs=(sent_locker,lock, sentence_counter))
        pool.map(self.extract_patterns_from_file, self.data_wrapper.ngrams_files)
        pool.close()
        pool.join()
        total_time = time.time()-startTime
        print "extract_patterns_matching_async running time: {}".format(total_time)
        # for file in self.data_wrapper.ngrams_files:
        #     self.extract_patterns_from_file(file)



    @abstractmethod
    def get_head_noun(self, after_pattern_idx, sentence, wildcard_indexes):
        return NO_HEAD_NOUN



    def extract_pattern_in_sentence(self,sentence):
        '''
        :param sentence:(SentenceData)
        :return:
        '''

        formatted_sentence = [Pattern.POS_TAG if word_data.pos in Pattern.POSSIBLE_TAGS
                              else word_data.word for word_data in sentence]
        # wildcard_indexes = [idx for idx,word_data in enumerate(sentence) if word_data.pos in Pattern.POS_TAG]
        for i in xrange(0,sentence.len-self.min_pattern_len+1):
            for j in xrange(i+self.min_pattern_len , min(i+self.max_pattern_len+1,sentence.len)):
                string = " ".join(formatted_sentence[i:j])
                if self.pattern_to_data.has_key(string):
                    matched_string = " ".join(sentence.words_sequence[i:j])
                    wildcard_indexes = [i+wc_idx for wc_idx in self.pattern_to_data[string].wildcards_idx]
                    head_noun = self.get_head_noun(j, sentence, wildcard_indexes)

                    matched_string  = "{} {}".format(matched_string, head_noun)
                    lock.acquire()
                    pattern_data = self.pattern_to_data[string]
                    pattern_data.add_matching_string(matched_string, sentence.occurrences)
                    self.pattern_to_data[string] = pattern_data
                    lock.release()
                    # self.pattern_to_data[string].add_matching_string(matched_string, sentence.occurrences)

                    break


    def export_results(self,output_folder):

        import time

        localtime = time.asctime( time.localtime(time.time()) )
        output_folder = output_folder+'_'+localtime
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for pattern in self.pattern_to_data.keys():
            pattern_data = self.pattern_to_data[pattern]
            file_name = os.path.join(output_folder, pattern_data.org_pattern)
            with open(file_name,'a') as f:
                rows = ['{}\t{}'.format(x[0],x[1]) for x in pattern_data.sorted_matching_strings]
                f.write("\n".join(rows))


class PatternDataHandlerNgrams(PatternDataHandler):

    def __init__(self,initial_patterns_list, input_text_file, data_source_type, read_file_format='rb'):
        super(PatternDataHandlerNgrams, self).__init__(initial_patterns_list, input_text_file, data_source_type, read_file_format=read_file_format)

    def get_head_noun(self, after_pattern_idx, sentence, wildcard_indexes):
        head_noun = NO_HEAD_NOUN
        if after_pattern_idx < sentence.len and sentence.pos_sequence[after_pattern_idx] in Pattern.NOUN_TAGS:
            head_noun = sentence.words_sequence[after_pattern_idx]
        return head_noun


class PatternDataHandlerHeadNoun(PatternDataHandler):


    def __init__(self,initial_patterns_list, input_text_file, data_source_type, read_file_format='rb'):
        super(PatternDataHandlerHeadNoun, self).__init__(initial_patterns_list, input_text_file, data_source_type, read_file_format=read_file_format)

    def get_head_noun(self, after_pattern_idx, sentence, wildcard_indexes):
        head_noun = NO_HEAD_NOUN
        for wildcard_idx in wildcard_indexes:
            head_idx = sentence[wildcard_idx].head
            if head_idx > -1:
                head_pos = sentence[head_idx].pos
                if head_pos in Pattern.NOUN_TAGS:
                    head_noun = sentence[head_idx].word
                    break
        return head_noun


def patterns_handlers_factory(handler_name,initial_patterns_list, input_text_file,data_source_type,read_file_format='rb'):
    if handler_name.lower() == "ngram":
        handler = PatternDataHandlerNgrams(initial_patterns_list, input_text_file,data_source_type,read_file_format=read_file_format)

    elif handler_name.lower() == "dep":
        handler = PatternDataHandlerHeadNoun(initial_patterns_list, input_text_file,data_source_type,read_file_format=read_file_format)

    return handler
