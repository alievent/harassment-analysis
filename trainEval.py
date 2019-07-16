from models.cnnModel import CNNModel
from models.jcnnModel import JCNNModel
from models.jacnnModel import JACNNModel
from models.jsacnnModel import JSACNNModel
from models.jabiLSTMModel import JABiLSTMModel


from configs.config import Config
from dataProcess.data import Data
from dataProcess.vocabulary import Vocabulary
from dataProcess.loadData import loadIds, load_extractions,load_forms,load_classifications,merge_data
from utilities.utils import loadWordVecfromText,segment_sentences_train

import pickle
import os
import gensim

if __name__ == '__main__':
    config = Config()
    if config.model == 'CNN':
        model = CNNModel(config)
    elif config.model == 'JABiLSTM':
        model = JABiLSTMModel(config)
    elif config.model == 'JCNN':
        model = JCNNModel(config)
    elif config.model == 'JACNN':
        model = JACNNModel(config)
    elif config.model == 'JSACNN':
        model = JSACNNModel(config)
    else:
       raise Exception
    if not os.path.exists(config.train_path):
        os.mkdir(config.train_path)

    trainData = Data()
    devData = Data()
    testData = Data()

    extraction_file = config.extraction_file
    classification_file = config.classification_file
    forms_file = config.forms_file

    trainIdFile =config.trainIdFile #'../data/trainIds.txt'
    devIdFile = config.devIdFile #'../data/testIds.txt'
    testIdFile = config.testIdFile

    train_ids = loadIds(trainIdFile)
    dev_ids = loadIds(devIdFile)
    test_ids = loadIds(testIdFile)


    doc_tokens, doc_tags = load_extractions(config.extraction_file) #key element extraction file
    doc_cls = load_classifications(config.classification_file) #class label file
    doc_forms = load_forms(config.forms_file) #harassform label file from previous work

    doc_tokens, doc_tags =segment_sentences_train(doc_tokens, doc_tags, config.delimiters)

    train_tokens, train_class_labels, train_tags = merge_data(doc_tokens, doc_tags, doc_cls, doc_forms,
                                                              config.cls_names, config.form_names, train_ids, config.filter)
    dev_tokens,dev_class_labels, dev_tags = merge_data(doc_tokens, doc_tags, doc_cls, doc_forms, config.cls_names,
                                                          config.form_names, dev_ids,config.filter)

    test_tokens,test_class_labels, test_tags = merge_data(doc_tokens, doc_tags, doc_cls, doc_forms, config.cls_names,
                                                          config.form_names, test_ids,config.filter)


    if config.pre_trained_wordVectors_path.endswith('.bin') :
        pretrained_word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(os.path.dirname(__file__), config.pre_trained_wordVectors_path), binary=True)
    else:
        pretrained_word_vectors = loadWordVecfromText(config.pre_trained_wordVectors_path)

    #included_tags = config.included_tags
    selected_posTag = config.selected_posTag
    #build vocabulary
    vocab = Vocabulary(config)
    vocab.buildVocabulary(train_tokens, selected_posTag)
    if config.max_num_words > 0:
        vocab.limit_vocab(config.max_num_words)
    pickle.dump(vocab, open(config.vocab_path, "wb"))
    vocab.loadWordVectors(pretrained_word_vectors)



    trainData.buildBasicData(train_tokens, train_class_labels, train_tags, vocab, config.cls_names, config.num_tags,
                             config.window_size, config.doc_max_length)
    devData.buildBasicData(dev_tokens, dev_class_labels, dev_tags, vocab, config.cls_names, config.num_tags,
                           config.window_size, config.doc_max_length)

    testData.buildBasicData(test_tokens, test_class_labels, test_tags, vocab, config.cls_names, config.num_tags,
                            config.window_size, config.doc_max_length)


    model.addVocab(vocab)

    model.buildModel()
    model.train(trainData.data,devData.data)

    config.model_path = model.config.model_path
    pickle.dump(config,open(config.config_path,'wb'))
    print(config.config_path)

    print('cls evaluations')
    testData.buildBasicData(test_tokens, test_class_labels, test_tags, vocab, config.cls_names, config.num_tags,
                            config.window_size, config.doc_max_length)
    cls_report, _ = model.eval(testData.data, config)
    print(cls_report)

    print('tag evaluation')

    testData.buildEvalData(test_tokens, test_class_labels, test_tags, vocab, config.cls_names, config.num_tags,
                          config.window_size, config.doc_max_length)
    _,tag_report = model.eval(testData.data,config)
    print(tag_report)
