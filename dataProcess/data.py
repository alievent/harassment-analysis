import numpy as np
from .encode import encode_in_sentence_train,encode_in_sentence_eval
class Data():
    def __init__(self):
        self.data = {}
        self.tag_distribution = []

    def buildBasicData(self,docs_input,doc_cls_input, labels_input,vocab,cls_names, num_tags, window_size,doc_max_length):
        data_doc,data_words, data_tag, data_doc_actual_length, tags_distr, data_events, data_sample_tk_map = encode_in_sentence_train(docs_input,
                                                                                       labels_input, vocab.word2idx, window_size,doc_max_length,num_tags)
        data_doc_actual_length = np.array(data_doc_actual_length)
        #print(data_x[0])
        data_doc = np.array(data_doc)
        data_words = np.array(data_words)
        data_tag = np.array(data_tag)

        #data_cls = np.array(doc_cls_input)
        #print(data_cls.shape)
        #self.sequence_length = data_x.shape[1]
        self.data ={'input_words':data_words,
                    'input_doc':data_doc,
               'input_tag':data_tag,
               #'input_cls':data_cls,
                'input_doc_actual_length':data_doc_actual_length,
                'input_events':np.array(data_events),
                'input_sample_tk_map':np.array(data_sample_tk_map)}
        for cls_names in cls_names:
            self.data[cls_names] =  np.array(doc_cls_input[cls_names])
        self.tag_distribution =tags_distr
        print(tags_distr)
    def buildEvalData(self,docs_input,doc_cls_input, labels_input,vocab,cls_names, num_tags, window_size,doc_max_length):
        data_doc,data_words, data_tag, data_doc_actual_length, doc_cls, tags_distr, data_events, data_sample_tk_map = encode_in_sentence_eval(docs_input,
                                                                                       labels_input, vocab.word2idx, window_size,doc_max_length,num_tags,doc_cls_input)
        data_doc_actual_length = np.array(data_doc_actual_length)
        #print(data_x[0])
        data_doc = np.array(data_doc)
        print('----')
        print(data_doc.shape)
        data_words = np.array(data_words)
        data_tag = np.array(data_tag)

        #data_cls = np.array(doc_cls)
        print("input words: ", data_words.shape)
        #self.sequence_length = data_x.shape[1]
        self.data ={'input_words':data_words,
                    'input_doc': data_doc,
                    'input_tag':data_tag,
               #'input_cls':data_cls,
                'input_doc_actual_length':data_doc_actual_length,
                'input_events':np.array(data_events),
                'input_sample_tk_map':np.array(data_sample_tk_map)}
        for cls_names in cls_names:
            self.data[cls_names] =  np.array(doc_cls[cls_names])
        self.tag_distribution =tags_distr
        print(tags_distr)

    def addData(self,key,dataValue):
        self.data[key] = dataValue