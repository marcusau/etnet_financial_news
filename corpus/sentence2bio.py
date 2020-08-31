import numpy as np
import pandas as pd
import re
from numpy import log, min
import os
import jieba_fast as jieba


import random
from datetime import date
import pickle
# 初始化[可选]

# 初始化时，可以指定自己的词典

from datetime import datetime, timedelta
from collections import defaultdict

#import openCC
from tqdm import tqdm
import time,json
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)


today_str=date.today().strftime('%Y-%m-%d').replace('-','')
#
# e_ner_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\ner_before_training'
# e_ner_filename='diva_finance.txt'
# e_ner_file=os.path.join(e_ner_folder,e_ner_filename)
# e_ner=pd.read_csv(e_ner_file,sep=' ',header=0,encoding='utf-8')['word'].values
#
#
sentences_folder='sentences'
NER_resources_folder='NER_resources'
bio_folder='bio_corpus'


ner_txtfilename='ner_results.txt'
ner_txtfile=os.path.join(NER_resources_folder,ner_txtfilename)
ners=pd.read_csv(ner_txtfile,sep=' ',names=['word','ner'],encoding='utf-8')

jieba.set_dictionary(os.path.join(NER_resources_folder,'userdict_ner.txt'))
jieba.load_userdict(os.path.join(NER_resources_folder,'userdict_ner.txt'))


sentence_filename='whole.txt'
sentence_file=os.path.join(sentences_folder, sentence_filename)



def convert_sentence_bio(sentence):


    word_list, label_list=[],[]

    for c in jieba.lcut(sentence.strip('\n'),cut_all=False,HMM=False):
        if c not in ners.word.values:
            c_tag='O'
            word_tag=list(zip(list(c), [c_tag]*len(c)))
        else:
            c_ner=ners[ners['word']==c]['ner'].values[0]
            if len(c)==1:
                c_tag=['I'+'-'+str(c_ner)]
            else:
                c_tag_head='B'+'-'+str(c_ner)
                c_tag_middle = ['I' + '-' + str(c_ner)]*(len(c)-2)
                c_tag_tail = 'I' + '-' + str(c_ner)
                c_tag=[c_tag_head]+c_tag_middle+[c_tag_tail]
            word_tag=list(zip(list(c), c_tag))

        word_list+=[i[0] for i in word_tag]
        label_list +=[ i[1] for i in word_tag]

    assert len(word_list)==len(label_list)

    record=list(zip(word_list,label_list))
    return record

def save_bio(news_txtfile):
    with open(news_txtfile, 'r', encoding='utf-8') as f:
            sentences = [s.strip().strip('\n').strip('*') for s in f.readlines()]


    sentences=[s for s in sentences if len(s)>4]
    num_sentences = len(sentences)
    print(num_sentences)
    random.shuffle(sentences)

    All_records = [convert_sentence_bio(sentence) for sentence in tqdm(sentences)]

    train_ratio = 0.85
    dev_ratio = 1-train_ratio

    train_split = int(len(All_records) * train_ratio)
    #dev_split = int(len(All_records) * (train_ratio + dev_ratio))

    train_records = All_records[:train_split]
    dev_records = All_records[train_split+1:] #dev_split
    #test_records = All_records[dev_split + 1:]

    with open(os.path.join(bio_folder,'train.txt'), 'w', encoding='utf-8') as f:
        for record in train_records:
            for i in record:
                f.write(i[0] + ' ' + i[1] + '\n')
            f.write('\n')

    with open(os.path.join(bio_folder,'dev.txt'), 'w', encoding='utf-8') as f:
        for record in dev_records:
            for i in record:
                f.write(i[0] + ' ' + i[1] + '\n')
            f.write('\n')

    # with open(os.path.join(corpus_folder,'test.txt'), 'w', encoding='utf-8') as f:
    #     for record in test_records:
    #         for i in record:
    #             f.write(i[0] + ' ' + i[1] + '\n')
    #         f.write('\n')




save_bio(sentence_file)

