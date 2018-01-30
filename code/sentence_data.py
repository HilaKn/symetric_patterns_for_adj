class WordData(object):

    def __init__(self,word,pos):
        self.word = word
        self.pos = pos

class SentenceData(object):

    def __init__(self,sentence_data):
        self.words_data_sequence = [WordData(word_data[0],word_data[1]) for word_data in sentence_data]
        self.words_sequence = [word_data.word for word_data in self.words_data_sequence]
        self.pos_sequence =  [word_data.pos for word_data in self.words_data_sequence]


    @property
    def len(self):
        return len(self.words_data_sequence)
