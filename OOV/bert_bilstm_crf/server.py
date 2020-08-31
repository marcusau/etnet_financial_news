#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import random
import re
import sys
import numpy as np
import pandas as pd

import json
import pickle
from tqdm import tqdm, trange

import datetime
import time

import sqlalchemy
from sqlalchemy import engine, VARCHAR
import mysql.connector


import torch


import jieba_fast as jieba
import jieba_fast.posseg as jieba_posseg

from  OOV.bert_bilstm_crf.models import BERT_BiLSTM_CRF
from  OOV.bert_bilstm_crf.server_util import text_preprocessing,DummySentencizer,tokens_to_spans
from transformers import (BertTokenizer,BertConfig)


print('\n\n')


#####----------------  Load NER data ----------------------------------------------------

today_str=datetime.date.today().strftime('%Y-%m-%d').replace('-','')
existing_ner_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\{}'.format(today_str)
#ner_folder=r'C:\Users\marcus\PycharmProjects\news_ner\seq_results'

print('loading NER txt file')
existing_ner_filename='ner_results.txt'
existing_ner_file= os.path.join(existing_ner_folder,existing_ner_filename)
existing_ner_df=pd.read_csv(existing_ner_file,sep=' ',names=['word','ner'])


#####----------------  Load jieba data ----------------------------------------------------
jieba_dict_folder=existing_ner_folder
jieba_dict_filename='userdict.txt'
jieba_dict_file=os.path.join(jieba_dict_folder,jieba_dict_filename)

print('loading jieba_dict')
jieba_dict_df=pd.read_csv(jieba_dict_file,sep=' ',names=['word','freq','pos'])
print('loading jieba.set_dictionary')
jieba.set_dictionary(jieba_dict_file)
print('loading jieba.load_userdict')
jieba.load_userdict(jieba_dict_file)

#####----------------  Load ner2pos data ----------------------------------------------------
ner2pos_folder=existing_ner_folder
ner2pos_filename='ner2pos.json'
ner2pos_file=os.path.join(ner2pos_folder,ner2pos_filename)

print('loading ner2pos')
with open(ner2pos_file, 'r') as fp:
    ner2pos=json.load(fp)

pos2ner={v:k for k,v in ner2pos.items()}
#print(pos2ner)

print('\n\n')
#####----------------  Load ner2pos data ----------------------------------------------------
model_dir = r'C:\Users\marcus\PycharmProjects\KG_test\OOV\bert_bilstm_crf/models'

device = 'cpu' #torch.device("cuda")


# -----------------------------------------------------------------------------------------------

# print('----------------Load Basic config files ------------------')

config_filename = 'config.json'
config_file = os.path.join(model_dir , config_filename)

training_args_filename = 'training_args.bin'
training_args_file = os.path.join(model_dir , training_args_filename)
training_args = torch.load(training_args_file)

label_list=training_args.label_list
num_labels = len(label_list)

label2id=training_args.label2id
id2label= training_args.id2label
max_seq_length = training_args.max_seq_length

# print('\n\n\n\n')
# print(' NER / OOV model config information')
# print(label_list)
# print('\n')
# print(num_labels )
# print('\n')
# print(id2label)
# print('\n')
# print(label2id)
# print('\n')
#
# print(training_args.need_birnn)
# print('\n')
# print(training_args.rnn_dim)
# print('\n')
# print(max_seq_length)
# print('\n\n')


# # # --------------------------------------------------------------------------
print(' loading NER model config')
config = BertConfig.from_pretrained(model_dir,     num_labels=num_labels,   id2label=id2label,  label2id=label2id, ) #
print(' loading NER (BERT) tokenizer')
tokenizer = BertTokenizer.from_pretrained(model_dir ,do_lower_case=training_args.do_lower_case,  use_fast=False)
print(' loading NER model ')
model = BERT_BiLSTM_CRF.from_pretrained(model_dir,config=config,  need_birnn=training_args.need_birnn, rnn_dim=training_args.rnn_dim, )
print(' loading NER model to {}'.format(device))
model.to(device)
model.eval()

#
def NER_model_predict(input_text):
    tokens = [w for word in list(input_text) for w in tokenizer.tokenize(word)]

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志


    ntokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
        input_mask.append(0)
    #
    # assert len(input_ids) == max_seq_length
    # assert len(segment_ids) == max_seq_length
    # assert len(input_mask) == max_seq_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)

    input_ids = input_ids.unsqueeze(0)
    segment_ids = segment_ids.unsqueeze(0)
    input_mask = input_mask.unsqueeze(0)

    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    input_mask = input_mask.to(device)

    pred_labels = []
    with torch.no_grad():
        logits = model.predict(input_ids, segment_ids, input_mask)
        # #             # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        # #             # logits = logits.detach().cpu().numpy()

        for l in logits:
            for idx in l:
               #print('idx:',idx,'id2label[idx]:',id2label[idx])
               pred_labels.append(id2label[idx])

    assert len(pred_labels) == len(ntokens)
   # print(ntokens, pred_labels)

    sentence_, spans = tokens_to_spans(ntokens, pred_labels,  allow_multiword_spans=True)

    return [(re.sub(' ', '', sentence_[span[0]:span[1]]), span[2]) for span in spans]


#if __name__ == '__main__':





def OOV_scan_run(sentences_list):
    # preprocessing_sentences =text_preprocessing(raw_text)
    #
    # preprocessing_sentences_list= DummySentencizer(preprocessing_sentences, split_characters=['。」' ,'。' ,';' ,'!' ,'*']).sentences
    #
    # preprocessing_sentences_list=[s for s in preprocessing_sentences_list if len(s)>3]

    OOV_scan_df = pd.DataFrame([], columns=['word', 'ner'])
    for sentence in sentences_list:

        ner_raw_results = NER_model_predict(sentence)  #python去除\ufeff、\xa0、\u3000
        ner_raw_results =[(ner[0], ner[1]) for ner in ner_raw_results if ner[1] not in ['J', 'UNIT', 'TIME', 'QUANTITY'] ]

        ## first round filter :  word with 'UNK' tokens
        ner_1stround_results=[]

        for ner_raw in ner_raw_results:
            word,ner_class=ner_raw[0],ner_raw[1]


            if re.findall(r'UNK',word):
                try:
                    word_unk_re=word.replace('[UNK]','\w')
                    word=re.findall(word_unk_re,sentence)[0]
                except Exception as e:
                    print('Cannot process UNK in:',sentence,word)

            ner_1stround_results.append((word,ner_class))

        ## second round filter : length > 1, word not exist in ner
        ner_2ndround_results = pd.DataFrame(ner_1stround_results, columns=['word', 'ner'])
        ner_2ndround_results = ner_2ndround_results[ner_2ndround_results['word'].str.len() > 1]
        ner_2ndround_results = ner_2ndround_results[~(ner_2ndround_results['word'].isin(existing_ner_df['word']))]
        ner_2ndround_results = ner_2ndround_results[~(ner_2ndround_results['word'].isin(jieba_dict_df['word']))]
        ner_2ndround_results = ner_2ndround_results[~(ner_2ndround_results['word'].isin(OOV_scan_df['word']))]


        ### OOV output by BERT-bilstm-crf model:
        if len(ner_2ndround_results) > 0:
            # print(sentence)
            #print(ner_2ndround_results)
            OOV_scan_df = pd.concat([OOV_scan_df, ner_2ndround_results], axis=0, ignore_index=True)

    return OOV_scan_df

def Autotag_gen(sentences_list):
    global existing_ner_df
    global jieba_dict_df

    OOVs_df = OOV_scan_run(sentences_list)
    #print(OOVs_df)
    #print('\n\n')
#### Post-processing for OOV:

###  OOV post-processing 1: adding OOVs to existing OOV data:
    existing_ner_df = pd.concat([existing_ner_df,OOVs_df], axis=0, ignore_index=True)

###  OOV post-processing 2: adding OOVs to jieba data:
    for i in OOVs_df.itertuples():
        OOV=str(i.word)
        OOV_nertag=str(i.ner)

        OOV_postag=ner2pos[OOV_nertag]
        jieba_freq = jieba.suggest_freq(OOV, tune=False)

        jieba.add_word(OOV, jieba_freq, tag=OOV_postag)
        jieba_dict_df = jieba_dict_df .append(pd.DataFrame([[OOV, jieba_freq,OOV_postag]], columns=jieba_dict_df .columns))

    del  OOVs_df
    #### Run Hash-tag generation

    # #### Hash-tag generation step 1: cleaning raw text + split into sentences
    # preprocessing_sentences =text_preprocessing(raw_sentences)
    # preprocessing_sentences_list= DummySentencizer(preprocessing_sentences, split_characters=['。」' ,'。' ,';' ,'!' ,'*']).sentences
    # preprocessing_sentences_list=[s for s in preprocessing_sentences_list if len(s)>1]

    #### Hash-tag generation step 2: tokenize the pre-processing sentences

    #### Hash-tag generation step 2.1 : apply jieba to tokenize the pre-processing sentences
    tokens=[]
    for preprocessing_sentence in sentences_list:
        tokens+=[(t,pos) for t,pos, in  jieba_posseg.cut(preprocessing_sentence,HMM=False) if pos  in ['nr','nt','ns','n','nz','x'] and len(t)>1]
        tokens=list(set(tokens))
        tokens = sorted(tokens, key=lambda x: len(x[0]), reverse=False)

#### Hash-tag generation step 2.2 : clean tokens
    tokens_clean=[]
    for index,(token,pos) in enumerate(tokens):
        sub_tokens=tokens[index+1:]
        Sum_check=0
        for sub_t,sub_pos in sub_tokens:
            if token in sub_t:
                Sum_check+=1
            else:
                Sum_check += 0

        if Sum_check==0:
            tokens_clean.append((token,pos))
    del tokens
# duplicate_check=pd.DataFrame(tokens_duplicate_check,columns=['token','pos','Sum_check'])
# duplicate_check.sort_values('Sum_check',ascending=True,inplace=True)
# duplicate_check.drop_duplicates('token',inplace=True)
# tokens_clean= list(duplicate_check.to_records(index=False))


    tokens_clean = sorted(tokens_clean, key=lambda x: len(x[0]), reverse=True)


    #### Hash-tag generation step 3: locate hash-tags by NER classes
    raw_autotags=[]
    for (token, postag) in tokens_clean:

        ##---- Hash-tag generation step 3.1 : Convert jieba POS tags into NER class
        if token in existing_ner_df['word'].values:

            ner_class = existing_ner_df[existing_ner_df['word']==token]['ner'].values[0]
            raw_autotags.append((token,ner_class))
       # print((token,ner_class))
        elif token in jieba_dict_df['word'].values:
        # print('cannot process: ',token,postag)
            try:
                ner_class = pos2ner[jieba_dict_df[jieba_dict_df['word'] == token]['pos'].values[0]]
                raw_autotags.append((token, ner_class))
            except:
                print(token,postag)
        #print((token, ner_class))
        # print(jieba_dict_df[jieba_dict_df['word']==token]['pos'].values[0])
        else:
        #print((token,postag))
            continue
        ##--- Hash-tag generation step 3.2 : filter of NER class
    del tokens_clean

    autotags=[]
    for (token,ner_class) in raw_autotags:

        if len(token) >= 5 and ner_class not in [ 'J']:
            autotags.append((token, ner_class))
        elif len(token) >2 and len(token)<5 and ner_class  not in ['TITLE','J']:
        #print(token,ner_class)
            autotags.append((token,ner_class))

        elif len(token) ==2 and ner_class  not in ['PRODUCT','TITLE','J','TERM']:
        #print(token,ner_class)
            autotags.append((token, ner_class))

    del raw_autotags

    autotags_df=pd.DataFrame(autotags,columns=['hashtags','type'])
    autotags_df.drop_duplicates('hashtags',inplace=True)
    return autotags_df



#------------------------------------------
#preprocess_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\{}'.format(today_str)
# master_dir= r'C:\Users\marcus\Desktop\etnet_2\trial_test\textdata'
#
# news_type='Total'
# news_folder=os.path.join(master_dir, news_type)
#
# Sentences_filename='Sentences.txt'
# Sentences_file=os.path.join(news_folder,Sentences_filename)
#
# newscontent_filename='newscontent.txt'
# newscontent_file=os.path.join(news_folder,newscontent_filename)
#
#
# topicID_filename='topcIDs.txt'
# topicID_file=os.path.join(news_folder,topicID_filename)
# topicIDs=tuple(int(id) for id in pd.read_csv(topicID_file,names=['topicID'])['topicID'].values)



### mysql setting -----------

mysql_host = '10.1.8.19'
mysql_port=3306
mysql_user = 'root'
mysql_password = 'abcd1234'
mysql_schema= 'etnet_financial_news'
mysql_tablename='news_test1'
#
#
mysql_conn = mysql.connector.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_schema,  auth_plugin='mysql_native_password')
mysql_cursor = mysql_conn.cursor()

sqlalchemy_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_schema}')
sqlalchemy_connection = sqlalchemy_engine.connect()


sql_select_query=f"""select newsID,headline, content from {mysql_tablename} where newsDate >='2020-03-15' and relCategory not like "%F1%" and relCategory not like "%L8%"  and relCategory not like "%L1%" and relCategory not like "%L11%"  and relCategory not like "%M9%";"""
mysql_cursor.execute(sql_select_query)


for row in tqdm(mysql_cursor.fetchall()[1536:]):
        newsID=row[0]
        headline=row[1]
        content=row[2]

        headline= text_preprocessing(headline)
        content = text_preprocessing(content)
        #
        # processed_content= ''.join([s for s in headline+[' ']+content if s !='*' ])
        # with open(newscontent_file, 'a', encoding='utf-8') as f:
        #     f.write(processed_content+'\n')


   # # sentences= DummySentencizer(content, split_characters=['。」','。',';','!','*']).sentences
        Sentences= [headline] +[' ']+ content#sentences

        Sentences= [s for s in Sentences if s != '*']
        Sentences = [s for s in Sentences if len(s) > 2]

        sentences_clean=[]
        for ss in Sentences:
            if len(ss)>128:
                ss = DummySentencizer(ss, split_characters=['，', '。', ';', '!', '*']).sentences
                for s in ss:
                    sentences_clean.append(s)
            else:
                sentences_clean.append(ss)

        autotags=Autotag_gen(sentences_clean)
        print(newsID)
        print(len(''.join(sentences_clean)))
        print(''.join(sentences_clean))
        autotags_list = autotags['hashtags'].values[:15]
        autotags_sql_text_format=str('\t'.join( autotags_list))
        print(autotags_sql_text_format)

        insert_autotags_query= "UPDATE {} SET autotags = '{}' WHERE newsID = {};".format(mysql_tablename,autotags_sql_text_format,newsID)
        mysql_cursor.execute(insert_autotags_query)
        mysql_conn.commit()

   # #
   # #



mysql_conn.close()
