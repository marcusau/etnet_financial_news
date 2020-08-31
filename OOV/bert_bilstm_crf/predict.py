#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import re
import sys
import numpy as np
import pandas as pd

import datetime
import time

import torch


import pickle

from tqdm import tqdm, trange

from  OOV.bert_bilstm_crf.models import Electra_BiLSTM_CRF,BERT_BiLSTM_CRF
from  OOV.bert_bilstm_crf.utils import text_preprocessing,DummySentencizer
from transformers import (ElectraConfig,ElectraTokenizer,BertTokenizer,BertConfig)


print('\n\n')


############################################################
today_str=datetime.date.today().strftime('%Y-%m-%d').replace('-','')
preprocess_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\{}'.format(today_str)
#ner_folder=r'C:\Users\marcus\PycharmProjects\news_ner\seq_results'

print('loading NER txt file')
ner_filename='ner_results.txt'
ner_file= os.path.join(preprocess_folder,ner_filename)
ner_results_df=pd.read_csv(ner_file,sep=' ',names=['word','ner'])


############################################################
# print('loading jieba.set_dictionary')
# jieba.set_dictionary(os.path.join(ner_folder,'userdict.txt'))
# print('loading jieba.load_userdict')
# jieba.load_userdict(os.path.join(ner_folder,'userdict.txt'))

print('loading jieba_dict_df')
jieba_dict_df=pd.read_csv(os.path.join(preprocess_folder,'userdict_ner.txt'),sep=' ',names=['word','freq','pos'])

# print('loading ner2pos')
# with open(os.path.join(ner_folder,'ner2pos.json'), 'r') as fp:
#     ner2pos=json.load(fp)
#
# pos2ner={v:k for k,v in ner2pos.items()}

# --------------------------------------------------------------------------
project_dir = r'C:\Users\marcus\PycharmProjects\KG_test\OOV\bert_bilstm_crf'
model_dir = os.path.join(project_dir,'models')

print('\n\n')
# --------------------------------------------------------------------------
device = 'cpu' #torch.device("cuda")

print('\n\n')
# -----------------------------------------------------------------------------------------------

# print('----------------Load Basic config files ------------------')

config_filename = 'config.json'
config_file = os.path.join(model_dir , config_filename)


training_args_filename = 'training_args.bin'
training_args_file = os.path.join(model_dir , training_args_filename)

# with open(label_list_file, 'rb') as f:
#     label_list = pickle.load(f)


# with open(os.path.join(model_dir, "label2id.pkl"), "rb") as f:
#     label2id = pickle.load(f)
#
#
# id2label = {value:key for key,value in label2id.items()}
# print(label2id)
# print(id2label )
# print('\n\n')

training_args = torch.load(training_args_file)

label_list=training_args.label_list
num_labels = len(label_list)

label2id=training_args.label2id
id2label= training_args.id2label
max_seq_length = training_args.max_seq_length


# with open('tag_to_id.json','w') as f:
#     json.dump(label2id,f)
# print('\n')
#print(config_file)
#print('\n')
# print(special_tokens_map)
# # print('\n')
print(label_list)
print('\n')
print(num_labels )
print('\n')
print(id2label)
print('\n')
print(label2id)
print('\n')

print(training_args.need_birnn)
print('\n')
print(training_args.rnn_dim)
print('\n')
print(max_seq_length)
print('\n\n')
# # #
# # # --------------------------------------------------------------------------
# config = ElectraConfig.from_pretrained(model_dir,     num_labels=num_labels)
# tokenizer = ElectraTokenizer.from_pretrained(model_dir)
# model = Electra_BiLSTM_CRF.from_pretrained(model_dir,config=config,  need_birnn=training_args.need_birnn, rnn_dim=training_args.rnn_dim, )
# model.to(device)


# # # --------------------------------------------------------------------------

config = BertConfig.from_pretrained(model_dir,     num_labels=num_labels,   id2label=id2label,  label2id=label2id, ) #
tokenizer = BertTokenizer.from_pretrained(model_dir ,do_lower_case=training_args.do_lower_case,  use_fast=False)
model = BERT_BiLSTM_CRF.from_pretrained(model_dir,config=config,  need_birnn=training_args.need_birnn, rnn_dim=training_args.rnn_dim, )
model.to(device)

# # # --------------------------------------------------------------------------
def tokens_to_spans(tokens, tags, allow_multiword_spans=False):
    """ Convert from tokens-tags format to sentence-span format. Some NERs
        use the sentence-span format, so we need to transform back and forth.
        Parameters
        ----------
        tokens : list(str)
            list of tokens representing single sentence.
        tags : list(str)
            list of tags in BIO format.
        allow_multiword_spans : bool
            if True, offsets for consecutive tokens of the same entity type are
            merged into a single span, otherwise tokens are reported as individual
            spans.
        Returns
        -------
        sentence : str
            the sentence as a string.
        spans : list((int, int, str))
            list of spans as a 3-tuple of start position, end position, and entity
            type. Note that end position is 1 beyond the actual ending position of
            the token.
    """
    spans = []
    curr, start, end, ent_cls = 0, None, None, None
    sentence = " ".join(tokens)
    if allow_multiword_spans:

         for token, tag in zip(tokens, tags):
             try:
                if tag == "O":
                    if ent_cls is not None:
                        spans.append((start, end, ent_cls))
                        start, end, ent_cls = None, None, None
                elif tag.startswith("B-"):
                    ent_cls = tag.split("-")[1]
                    start = curr
                    end = curr + len(token)
                else:  # I-xxx
                    try:
                         end += len(token) + 1
                    except:
                         end=0
                         end += len(token) + 1
            # advance curr
                curr += len(token) + 1
             except Exception as e:
                print('cannot process:', token, tag,'reason',e)
            # handle remaining span
         if ent_cls is not None:
            spans.append((start, end, ent_cls))


    else:
        for token, tag in zip(tokens, tags):
            if tag.startswith("B-") or tag.startswith("I-"):
                ent_cls = tag.split("-")[1]
                start = curr
                end = curr + len(token)
                spans.append((start, end, ent_cls))
            curr += len(token) + 1

    return sentence, spans

#
def model_predict(input_text):
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

    # encoded_tokens = tokenizer.encode_plus(input_text, max_length=max_seq_length, add_special_tokens=True,     pad_to_max_length=True)
    # input_ids = encoded_tokens['input_ids']
    # segment_ids = encoded_tokens['token_type_ids']
    # input_mask = encoded_tokens['attention_mask']
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


if __name__ == '__main__':

    new_ner = pd.DataFrame([], columns=['word', 'ner'])

    news_type = 'Total'
    news_folder = os.path.join(r'C:\Users\marcus\Desktop\etnet_2\trial_test\textdata', news_type)
  #  sentences_file = os.path.join(news_folder, news_type + '.txt')
    sentences_file = os.path.join(news_folder,  'Sentences.txt')


    with open(sentences_file, 'r', encoding='utf-8') as f:
        preprocessing_sentences_list = [s for s in f.readlines() if len(s)>4]
#
#     # #input_text = '香港作曲家及作詞家協會(CASH)今日向會員發出一次過5000元特別版權費，協助他們渡過疫情難關。'
#
#     raw_news='''《經濟通通訊社１２日專訊》外電報道，內地電商唯品會考慮來港作第二上市，公司現正與
# 財務顧問就上市進行磋商，初步預計最快於今年內上市。
# 　　據唯品會向美國證交會提交文件顯示，截至今年３月底，騰訊（００７００）旗下
# Ｔｅｎｃｅｎｔ　Ｍｏｂｉｌｉｔｙ　Ｌｉｍｉｔｅｄ持有唯品會約９﹒６％，京東
# （０９６１８）旗下ＪＤ　Ｅｎｔｉｔｉｅｓ持股約７﹒５％。
# 　　唯品會於２００８年８月成立，公司主要業務為互聯網在線銷售品牌折扣商品，當中包括名
# 品服飾鞋包、美妝、居家等產品。（ｂｉ）'''
#
#     raw_sentences  =re.sub('\n|\s' ,'' ,raw_news)
#     print('\n')
#     preprocessing_sentences =text_preprocessing(raw_sentences)
#
#     preprocessing_sentences_list= DummySentencizer(preprocessing_sentences, split_characters=['。」' ,'。' ,';' ,'!' ,]).sentences


    for sentences_id,input_text_clean in  tqdm(enumerate(preprocessing_sentences_list[:])):
    #
           # input_text_clean=input_text.replace('\u3000', '').replace('\ufeff', '').replace('\xa0', '')
            if len(input_text_clean)>3:
            #print(input_text_clean)

                ner_results=model_predict(input_text_clean)  #python去除\ufeff、\xa0、\u3000
    #
                ner_result_raw=[(ner[0], ner[1]) for ner in ner_results if ner[1] not in ['J', 'UNIT', 'TIME', 'QUANTITY'] ]

                ner_results=[]
                for ner_raw in ner_result_raw:
                    word,ner_class=ner_raw[0],ner_raw[1]
                    if re.findall(r'UNK',word):
                        try:
                            word_unk_re=word.replace('[UNK]','\w')
                            word=re.findall(word_unk_re,input_text_clean)[0]
                        except Exception as e:
                            print('Cannot process UNK in:',input_text_clean,word)
                    ner_results.append((word,ner_class))

                ner_finding_df = pd.DataFrame(ner_results,columns=['word', 'ner'])

                ner_finding_df = ner_finding_df[ner_finding_df['word'].str.len() > 1]

                ner_finding_df = ner_finding_df[~(ner_finding_df['word'].isin(ner_results_df['word']))]
                ner_finding_df = ner_finding_df[~(ner_finding_df['word'].isin(jieba_dict_df['word']))]
                ner_finding_df = ner_finding_df[~(ner_finding_df['word'].isin(new_ner['word']))]

                if len(ner_finding_df) > 0:
                #print(input_text)
                    print(ner_finding_df)
                    new_ner = pd.concat([new_ner, ner_finding_df], axis=0, ignore_index=True)

                #print('\n')
    ner_results_df = pd.concat([ner_results_df,new_ner], axis=0, ignore_index=True)

    # for i in new_ner.itertuples():
    #     OOV=str(i.word)
    #     OOV_nertag=str(i.ner)
    #     OOV_postag=ner2pos[OOV_nertag]
    #     jieba_freq = jieba.suggest_freq(OOV, tune=False)
    #     jieba.add_word(OOV, jieba_freq, tag=OOV_postag)
    #
    # tokens=[]
    # for sentences_id, input_text in tqdm(enumerate(preprocessing_sentences_list[:])):
    #     tokens+=[(t,pos) for t,pos, in jieba.posseg.cut(input_text,HMM=False) if pos  in ['nr','nt','ns','n','nz','x'] and len(t)>1]
    # tokens=list(set(tokens))
    # tokens = sorted(tokens, key=lambda x: len(x[0]), reverse=False)
    #
    #
    #
    # tokens_clean=[]
    # for index,(token,pos) in enumerate(tokens):
    #
    #     sub_tokens=tokens[index+1:]
    #
    #     Sum_check=0
    #     for sub_t,sub_pos in sub_tokens:
    #         if token in sub_t:
    #             Sum_check+=1
    #     if Sum_check==0:
    #         tokens_clean.append((token,pos))
    #
    #
    # tokens_clean = sorted(tokens_clean, key=lambda x: len(x[0]), reverse=True)
    #
    # print('新聞原文:')
    # for s in preprocessing_sentences_list:
    #     print(s)
    #
    # print('\n\n')
    #
    #
    # print('hashtags:')
    #
    # for token,postag in tokens_clean:
    #    try:
    #         nertag=ner_results_df[ner_results_df['word']==token]['ner'].values[0]
    #
    #    except:
    #        #print('cannot process: ',token,postag)
    #         nertag=pos2ner[jieba_dict_df[jieba_dict_df['word']==token]['pos'].values[0]]
    #
    #    if len(token) >2 and nertag  not in ['TITLE','J']:
    #        print(token)
    #       # print(token,nertag)
    #
    #    elif len(token) ==2 and nertag  not in ['PRODUCT','TITLE','J']:
    #        print(token)
          # print(token,nertag)


    #     # except Exception as e:
    #     #     print('Error',e,' cannot process:', sentences_id, ' ',input_text)
    #     # with open(os.path.join(ner_folder, 'fail.txt'),'a',encoding='utf-8') as f:
    #     #     f.write(input_text)
    # # # #
    # # # #
    # # # # #new_ner.word.str.len().sort_values(ascending=False)



    new_ner.drop_duplicates('word', inplace=True)
    index_sorted = new_ner.word.str.len().sort_values(ascending=False).index
    new_ner = new_ner.reindex(index_sorted)
    new_ner = new_ner.reset_index(drop=True)


    new_ner_filename='ner_new_{}.txt'.format(news_type.replace('.txt',''))
    new_ner.to_csv(os.path.join(preprocess_folder,new_ner_filename ), sep=' ', index=None, header=0, encoding='utf-8')