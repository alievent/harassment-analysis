import csv
import os
import random
import re
from utilities.constant import tagMap
import json

def process_text(str):
    str = str.replace("."," . ")
    str = str.replace(","," , ")
    str = str.replace("!"," ! ")
    str = str.replace("?"," ? ")
    str = str.replace("quot", ' " ')
    str = re.sub(r"[^A-Za-z0-9(),!?.]", " ", str)
    return str.lower()

def annotation_parser(records,id):
    annotations = {}
    annStarts = {}
    annEnds ={}
    count = 0
    overlap = False
    for type in records:
        for tag in records[type]:
            aid = count
            start = int(tag['offset'][0])
            end = int(tag['offset'][1])
            annotations[aid] = [type.lower(),start,end,tag['span']]
            annStarts[start] = aid
            if end in annEnds:
                print("end exists",id)
                overlap = True
            annEnds[end] = aid
            count += 1
    return annotations, annStarts, annEnds,overlap

def processDoc(origTxt,annotations, annStarts, annEnds,id):
    tokens = []
    tags = []
    i = 0
    current = 0
    meetStart = False
    overlap = False
    for i in range(len(origTxt)):


        if i in annEnds:
            meetStart = False

            aId = annEnds[i]
            start = annotations[aId][1]
            end = annotations[aId][2]
            type = annotations[aId][0]
            processed_text = process_text(origTxt[start:i])
            splits = processed_text.split()
            tokens += splits
            tags += [tagMap[type] for _ in range(len(splits))]
            assert origTxt[start:end] == origTxt[start:i]
            current = i
        if i in annStarts:
            if meetStart:
                print(id,i)
                overlap = True
            else:
                meetStart = True
            processed_text = process_text(origTxt[current:i])
            splits = processed_text.split()
            tokens += splits
            tags += [0 for _ in range(len(splits))]
            current = i

    i = len(origTxt)
    if i in annEnds:

        aId = annEnds[i]
        start = annotations[aId][1]
        end = annotations[aId][2]
        type = annotations[aId][0]
        processed_text = process_text(origTxt[start:i])
        splits = processed_text.split()
        tokens += splits
        tags += [tagMap[type] for _ in range(len(splits))]
        assert origTxt[start:end] == origTxt[start:i]
        current = i

    processed_text = process_text(origTxt[current:])
    splits = processed_text.split()
    tokens += splits
    tags += [0 for _ in range(len(splits))]

    assert len(tokens) == len(tags)
    return tokens, tags,overlap

def load_extractions(input_path, filter_overlap=True):
    doc_tokens = {}
    doc_tags = {}
    #corpusTags = {}
    for filename in os.listdir(input_path):
        #print(filename)
        if filename.startswith("B"):
            #print(filename)
            docs_raw = json.load(open(input_path + filename,'r'))

            for id in docs_raw:
                doc_id = int(docs_raw[id]['meta']['originalName'][:-4])

                txt =docs_raw[id]['content']
                #print(fileId)
                annotations, annStarts, annEnds, overlap = annotation_parser(docs_raw[id]['records'],filename + '\t'+str(doc_id))


                tokens, tags,overlap =processDoc(txt, annotations, annStarts, annEnds,filename + '\t' +str(doc_id))
                if overlap and filter_overlap:
                    # print(id)
                    continue
                doc_tokens[doc_id]=tokens
                doc_tags[doc_id]=tags
    return doc_tokens, doc_tags

def process_cls(label):
    return [int(x) for x in label.split(';')]

def load_classifications(data_file):
    docs = {}
    with open(data_file, 'r', newline='') as df:
        reader = csv.reader(df, delimiter=',')

        count = 0
        for row in reader:
            #print(row)
            id = row[0]
            #print(len(id))
            age = row[1]
            num = row[2]
            harasser_type = row[3]
            location_type = row[4]
            time = row[5]
            count += 1

            if count == 1 or len(num)== 0 or num is None or age =='无效数据':
                continue
            id = int(id)
            docs[id] = {}
            docs[id]['harasser_age'] = process_cls(age)
            docs[id]['harasser_num'] = process_cls(num)
            docs[id]['harasser_type'] = process_cls(harasser_type)
            docs[id]['location_type'] = process_cls(location_type)
            docs[id]['time_of_day'] = process_cls(time)

    return docs

def load_forms(data_file):
    docs = {}
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        i = -1
        for row in reader:

            if i >= 0:
                id = int(row[0])
                description = row[1]
                commenting = int(row[2])
                ogling = int(row[3])
                touching = int(row[4])
                docs[id] = {}
                #corpus[id]['description'] = description
                docs[id]['commenting'] = [int(commenting)]
                docs[id]['ogling'] = [int(ogling)]
                docs[id]['touching'] = [int(touching)]
            i += 1
    return docs




def loadIds(idFile):
    ids =[]

    with open(idFile, 'r') as idf:
        for line in idf.readlines():
            ids.append(int(line.strip()))
    return ids

def tokenizeData(data_input):
    tokenizedData=[]
    for doc in data_input:
        tokenizedData.append(doc.split())
    return tokenizedData



def checkDocAndTokens(docs, tokens):
    for docId in tokens:
        tks = docs[docId]['text'].split()
        for i, tk in enumerate(tks):
            if tk != tokens[docId][i]:
               print("miss match docId : " + str(docId))
               break
        break
def merge_data(docs_tokens, docs_tags, doc_cls, doc_forms, cls_names, form_names,ids,filter = False):
    tokens = []

    tags = []
    class_labels={}
    for cls in cls_names:
        class_labels[cls] = []

    missing_ids = []
    for docId in ids:
        if docId not in docs_tags or \
            docId not in doc_cls or  \
            docId not in doc_forms:
            missing_ids.append(docId)
            continue
        labels = {}
        tmp = False
        for cls in cls_names:
            if cls in form_names:
                labels[cls] = doc_forms[docId][cls]
            else:
                labels[cls] = doc_cls[docId][cls]
                if len(labels[cls]) > 1:
                    tmp = True


        if filter and tmp:
            print(str(docId) + ' filtered')
            continue
        tokens.append(docs_tokens[docId])
        for cls in cls_names:
            class_labels[cls].append(labels[cls][0])
        tags.append(docs_tags[docId])
    print('missing ids', len(missing_ids))
    return tokens, class_labels, tags

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