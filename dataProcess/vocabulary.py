from utilities.constant import UNK, PAD, POS_UNK, POS_PAD
import numpy as np
import nltk
class Vocabulary(object):
    def __init__(self,config):
        self.word2idx = {}
        self.idx2word = {}
        self.word2freq = {}
        self.size = 0
        self.add_word(UNK)
        self.add_word(PAD)
        self.pos2idx = {}
        self.idx2pos = {}
        self.pos2freq = {}
        self.pos_size = 0
        self.add_pos(POS_UNK)
        self.add_pos(POS_PAD)
        self.config = config
        self.wordVectors = None

    def has(self, word):
        return word in self.word2idx

    def add_word(self, word):
        if not self.has(word):
            ind = self.size
            self.word2idx[word] = ind
            self.idx2word[ind] = word
            self.word2freq[word] = 1
            self.size += 1
        else:
            self.word2freq[word] += 1

    def to_idx(self, word):
        if self.has(word):
            return self.word2idx[word]
        else:
            return self.word2idx[UNK]

    def to_word(self, ind):
        if ind >= self.size:
            return 0
        return self.idx2word[ind]

    def to_freq(self, word):
        if self.has(word):
            return self.word2freq[word]
        else:
            return 0


    def pos_has(self, pos):
        return pos in self.pos2idx

    def add_pos(self, pos):
        if not self.pos_has(pos):
            ind = self.pos_size
            self.pos2idx[pos] = ind
            self.idx2pos[ind] = pos
            self.pos2freq[pos] = 1
            self.pos_size += 1
        else:
            self.pos2freq[pos] += 1

    def pos_to_idx(self, pos):
        if self.pos_has(pos):
            return self.pos2idx[pos]
        else:
            return self.pos2idx[POS_UNK]

    def pos_to_pos(self, ind):
        if ind >= self.pos_size:
            return 0
        return self.idx2pos[ind]

    def pos_to_freq(self, pos):
        if self.pos_has(pos):
            return self.pos2freq[pos]
        else:
            return 0

    def limit_vocab(self, max_vocab_size):
        if self.size > max_vocab_size:
            sorted_freq = sorted(self.word2freq.items(),
                                 key=lambda x: x[1], reverse=True)
            abandon_tokens = [t[0] for t in sorted_freq[max_vocab_size:]
                              if t[0] != UNK and t[0] != PAD]
            for token in abandon_tokens:
                self.word2freq.pop(token)
            left_words = set(self.word2idx) - set(abandon_tokens)
            self.word2idx = {}
            self.idx2word = {}
            cnt = 0
            for word in left_words:
                self.word2idx[word] = cnt
                self.idx2word[cnt] = word
                cnt += 1
            self.size = len(self.idx2word)

    def load_bin_vec(self, vocab_bin):
        word_vecs = np.zeros((len(self.word2idx), self.config.word_embedding_size))
        count = 0

        for word in self.word2idx:
            if vocab_bin is not None and word in vocab_bin:
                count += 1
                word_vecs[self.word2idx[word]] = (vocab_bin[word])
            else:
                word_vecs[self.word2idx[word]] = (np.random.uniform(-0.25, 0.25, self.config.word_embedding_size))
        return word_vecs

    def loadWordVectors(self, vocab_bin):

        self.wordVectors = self.load_bin_vec(vocab_bin)

    def buildVocabulary(self, docs,selected_posTag=None):
        for doc in docs:
            for sent in doc:
                for tk in sent:
                    self.add_word(tk)
                for pos in nltk.pos_tag(sent):
                    posTag = pos[1]
                    if posTag in selected_posTag or len(selected_posTag) == 0 or not selected_posTag:
                        self.add_pos(posTag)
                    else:
                        self.add_pos(POS_UNK)


