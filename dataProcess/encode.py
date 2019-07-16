from utilities.constant import UNK, PAD, POS_PAD, POS_UNK
import numpy as np
import nltk

def encode_in_sentence_train(docs, anchors, vocab, window_size,doc_max_length,num_tags):
    encoded_doc,doc_windows,windows, window, doc_labels,labels, labels_distr, doc_actual_length= [],[],[], [], [], [],[],[]
    #print len(docs)
    #print len(anchors)
    labels_distr = [0]*num_tags
    events = []
    sample_tk_map = []
    j = 0
    for doc in docs:
        s = 0
        #print len(doc)
        #print len(anchors[j])
        doc_events = []
        for sent in doc:
            #print len(sent)
            #posTags = nltk.pos_tag(sent)
            sent_events = []
            if len(sent) == 0:
                print("sent 0 length")
            for tok in np.arange(len(sent)):
                for i in np.arange(-window_size, window_size + 1):
                    if i + tok < 0 or i + tok >= len(sent):
                        window.append(vocab[PAD])
                        #posTag_window.append(posVocab[POS_PAD])
                    else:
                        if sent[i + tok] in vocab:
                            window.append(vocab[sent[i + tok]])
                        else:
                            window.append(vocab[UNK])
                        '''
                        if posTags[i + tok][1] in posVocab:
                            posTag_window.append(posVocab[posTags[i + tok][1]])
                        else:
                            posTag_window.append(posVocab[POS_UNK])
                        '''

                windows.append(window)
                #posTag_windows.append(posTag_window)
                labels.append(anchors[j][s][tok])
                labels_distr[anchors[j][s][tok]]+=1


                #print "j: " + str(j)
                #if j == 13:
                   #print (j,s,tok)
                   #print sent
                sample_tk_map.append((j,s,tok))
                if anchors[j][s][tok] != 0 :
                    if len(sent_events) > 0 and sent_events[-1][1] == anchors[j][s][tok] and sent_events[-1][2][-1] == tok - 1:
                        sent_events[-1][2].append(tok)
                    else:
                        sent_events.append((s,anchors[j][s][tok],[tok]))

                window = []
                #posTag_window=[]
            doc_events.append(sent_events)

            s +=1
        doc_actual_length.append(min(len(labels),doc_max_length))
        if len(windows) >= doc_max_length:
            windows = windows[:doc_max_length]
            labels = labels[:doc_max_length]
        else:
            windows+=[[vocab[PAD] for _ in range(2*window_size + 1)] for _ in range(doc_max_length - len(windows))]
            #print(len(windows))
            labels+=[0]*(doc_max_length - len(labels))
        doc_windows.append(windows)
        encoded_doc.append([window[window_size] for window in windows])
        windows =[]
        doc_labels.append(labels)
        labels = []
        events.append(doc_events)
        j += 1
    #print sample_tk_map
    return encoded_doc,doc_windows, doc_labels, doc_actual_length, labels_distr,events, sample_tk_map

def encode_in_sentence_eval(docs,anchors, vocab, window_size,doc_max_length,num_tags,doc_cls_input):
    encoded_doc,doc_windows,windows, window, doc_labels,labels, labels_distr, doc_actual_length= [],[],[], [], [], [],[],[]
    doc_cls ={}
    for cls_name in doc_cls_input:
        doc_cls[cls_name]=[]
    #print len(docs)
    #print len(anchors)
    labels_distr = [0]*num_tags
    events = []
    sample_tk_map = []
    num_words = 0
    #num_tags=0
    j = 0
    for doc in docs:
        s = 0
        #print len(doc)
        #print len(anchors[j])
        doc_events = []
        for sent in doc:
            #print len(sent)
            #posTags = nltk.pos_tag(sent)
            sent_events = []
            if len(sent) == 0:
                print("sent 0 length")
            for tok in np.arange(len(sent)):
                num_words+=1
                for i in np.arange(-window_size, window_size + 1):
                    if i + tok < 0 or i + tok >= len(sent):
                        window.append(vocab[PAD])
                        #posTag_window.append(posVocab[POS_PAD])
                    else:
                        if sent[i + tok] in vocab:
                            window.append(vocab[sent[i + tok]])
                        else:
                            window.append(vocab[UNK])
                        '''
                        if posTags[i + tok][1] in posVocab:
                            posTag_window.append(posVocab[posTags[i + tok][1]])
                        else:
                            posTag_window.append(posVocab[POS_UNK])
                        '''

                windows.append(window)
                #posTag_windows.append(posTag_window)
                labels.append(anchors[j][s][tok])
                labels_distr[anchors[j][s][tok]]+=1


                #print "j: " + str(j)
                #if j == 13:
                   #print (j,s,tok)
                   #print sent
                sample_tk_map.append((j,s,tok))
                if anchors[j][s][tok] != 0 :
                    if len(sent_events) > 0 and sent_events[-1][1] == anchors[j][s][tok] and sent_events[-1][2][-1] == tok - 1:
                        sent_events[-1][2].append(tok)
                    else:
                        sent_events.append((s,anchors[j][s][tok],[tok]))

                window = []
                #posTag_window=[]
            doc_events.append(sent_events)

            s +=1
        #doc_actual_length.append(min(len(labels),doc_max_length))

        if len(windows) < doc_max_length:
            doc_actual_length.append(len(windows))

            windows+=[[vocab[PAD] for _ in range(2*window_size + 1)] for _ in range(doc_max_length - len(windows))]
            labels+=[0]*(doc_max_length - len(labels))
            doc_windows.append(windows)
            doc_labels.append(labels)
            for cls_name in doc_cls_input:
                doc_cls[cls_name].append(doc_cls_input[cls_name][j])
            encoded_doc.append([window[window_size] for window in windows])

            windows = []
            labels = []


        while len(windows) > doc_max_length:
            new_windows = windows[:doc_max_length]
            new_labels = labels[:doc_max_length]
            encoded_doc.append([window[window_size] for window in new_windows])
            doc_windows.append(new_windows)
            doc_labels.append(new_labels)
            windows = windows[doc_max_length:]
            labels = labels[doc_max_length:]
            for cls_name in doc_cls_input:
                doc_cls[cls_name].append(doc_cls_input[cls_name][j])
            doc_actual_length.append(doc_max_length)
        if len(windows) > 0:
            doc_actual_length.append(len(windows))
            windows+=[[vocab[PAD] for _ in range(2*window_size + 1)] for _ in range(doc_max_length - len(windows))]
            labels+=[0]*(doc_max_length - len(labels))
            doc_windows.append(windows)
            encoded_doc.append([window[window_size] for window in windows])
            doc_labels.append(labels)
            for cls_name in doc_cls_input:
                doc_cls[cls_name].append(doc_cls_input[cls_name][j])
        windows = []
        labels = []
        events.append(doc_events)
        j += 1

    return encoded_doc,doc_windows, doc_labels, doc_actual_length, doc_cls,labels_distr,events, sample_tk_map