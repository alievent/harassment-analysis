import numpy as np
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

def data_iterator(orig_data, batch_size=50, shuffle=True):
    data = orig_data
    for key in data:
        data[key] = np.array(data[key])
    if shuffle:
        indices = np.random.permutation(len(data['input_doc']))
        for key in data:
            data[key] = data[key][indices]


    total_steps = int(np.ceil(len(data['input_doc']) / float(batch_size)))

    for step in range(total_steps):
        batch = {}
        batch_start = step * batch_size
        for key in data:
            batch[key] = data[key][batch_start:batch_start + batch_size]
        yield batch

def loadWordVecfromText(wv_path):
    wv ={}
    with open(wv_path,'r') as f:
        line = f.readline()
        word_num, vec_dim = line.split(" ")
        vec_dim = int(vec_dim)

        for i, line in enumerate(f):
            word = line.split(" ")[0]
            vec = [float(t) for t in line.split(" ")[1:-1]]
            assert len(vec) == vec_dim
            if word not in wv:
                wv[word] = vec
    return wv


def processBatchResultsForCls(doc_actual_length, predictions,doc_max_length):
    prev_doc_len = 0
    docs_tags = []
    for l in doc_actual_length:
        tag = predictions[prev_doc_len:prev_doc_len + l]
        if l < doc_max_length:
            tag += [0]*(doc_max_length - l)
        docs_tags.append(tag)
        prev_doc_len += l
    return docs_tags

def segment_sentences_train(docs, labels, delimiters):
    segmented_docs = {}
    segmented_labels ={}
    for id in docs:
        doc = docs[id]
        label = labels[id]
        sent = []
        sent_label = []
        segmented_doc = []
        segmented_label = []
        for j in range(len(doc)):
            tk = doc[j]
            l = label[j]
            sent.append(tk)
            sent_label.append(l)

            if tk in delimiters:
                segmented_doc.append(sent)
                segmented_label.append(sent_label)
                assert len(sent) == len(sent_label)
                sent =[]
                sent_label =[]
        if len(sent) > 0 :
            segmented_doc.append(sent)
            segmented_label.append(sent_label)
        assert len(segmented_doc) == len(segmented_label)
        segmented_docs[id]= segmented_doc
        segmented_labels[id]=segmented_label
    return segmented_docs,segmented_labels

def evalReport(y_true, y_pred,config,num_classes):
    micro_precision, micro_recall, micro_f1_score, status = precision_recall_fscore_support(y_true,
                                                                                            y_pred,
                                                                                            labels=range(0,
                                                                                                         num_classes),
                                                                                            pos_label=None,
                                                                                            average='micro')
    macro_precision, macro_recall, macro_f1_score, status = precision_recall_fscore_support(y_true,
                                                                                            y_pred,
                                                                                            labels=range(0,
                                                                                                         num_classes),
                                                                                            pos_label=None,
                                                                                            average='macro')

    cls_precision, cls_recall, cls_f1_score, status = precision_recall_fscore_support(y_true,
                                                                                      y_pred,
                                                                                      labels=range(0,
                                                                                                   num_classes),
                                                                                      pos_label=None,
                                                                                      average=None)
    accuracy = accuracy_score(y_true, np.array(y_pred), normalize=True)

    report = '\nconfigs: \n' + config.hyperParamToStr() \
             + "micro_precision : %f, micro_recall %f, micro_f1_score %f \n" % (
        micro_precision, micro_recall, micro_f1_score) \
             +"macro_precision : %f, macro_recall %f, macro_f1_score %f \n" % (
        macro_precision, macro_recall, macro_f1_score) \
             + "p r f for each cls :\n" + str(cls_precision) + '\n' + str(cls_recall) + '\n' + str(cls_f1_score)+'\n' \
             + 'accuracy: %f ' % accuracy


    return report
