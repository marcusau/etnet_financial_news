#/usr/bin/env python
# -*- coding: utf-8 -*-
#
# from fastHan import FastHan
# model=FastHan(model_type="large")

import os
import pandas as pd
import swifter
from tqdm import tqdm

import re
from full_width_to_half_width import full_width_to_half_width
from opencc import OpenCC
cc1 = OpenCC('hk2s')
cc2=OpenCC('s2hk')
import operator

from urlextract import URLExtract
url_extractor = URLExtract()

from datetime import datetime ,date, timedelta
from dateutil.relativedelta import relativedelta
import time


import sqlalchemy
from sqlalchemy import engine, VARCHAR
import mysql.connector

today_str=date.today().strftime('%Y-%m-%d').replace('-','')
#
# e_ner_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\ner_before_training'
# e_ner_filename='diva_finance.txt'
# e_ner_file=os.path.join(e_ner_folder,e_ner_filename)
# e_ner=pd.read_csv(e_ner_file,sep=' ',header=0,encoding='utf-8')['word'].values
#
#


### mysql setting -----------

mysql_host = '127.0.0.1'
mysql_port=3306
mysql_user = 'root'
mysql_password = '12341234'
mysql_schema= 'etnet_news'
mysql_tablename='news'
#
#
mysql_conn = mysql.connector.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_schema,  auth_plugin='mysql_native_password')
mysql_cursor = mysql_conn.cursor()

sqlalchemy_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_schema}')
sqlalchemy_connection = sqlalchemy_engine.connect()
###---- articles extraction --------------------



###----- text pre-processing --------------
delimiters=re.compile('，|。')

class DummySentencizer:
   def __init__(self, input_text, split_characters=['，','。'], delimiter_token='<SPLIT>'):
        self.sentences = []
        self.raw = str(input_text)
        self._split_characters=split_characters
        self._delimiter_token=delimiter_token
        self._index=0
        self._sentencize()

   def _sentencize(self):
       work_sentence = self.raw
       for character in self._split_characters:
           work_sentence = work_sentence.replace(character, character + "" + self._delimiter_token)
       self.sentences = [x.strip() for x in work_sentence.split(self._delimiter_token) if x != '']


def text_preprocessing(s):
    s1=DummySentencizer(s,split_characters=['，','。','*',';','!']).sentences
    s2=''.join([full_width_to_half_width(s[:-1])+s[-1] for s in s1])
    s2 = re.sub('﹒', '.', s2)
    s2 = re.sub(re.compile('https?://\S+'), '', s2)  ## ----remove html
    s2 = re.sub(re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), '',
                s2)  ## ----remove html
    s2 = re.sub(re.compile('\(\d{3,6}\)'), '', s2)
    s2 = re.sub(re.compile('(\(滬:\d+\))'), '', s2)
    s2 = re.sub(re.compile('(\(深:\d+\))'), '', s2)
    s2 = re.sub(re.compile('\(\d{3,6}\)'), '', s2)
    s2 = re.sub(re.compile('(^\《\w+\》)'), '', s2)
    s2 = re.sub(re.compile('(\《香港經濟日報\》$)'), '', s2)
    s2 = re.sub(re.compile('(\《環富通基金頻道\d+專訊\》)'), '', s2)
    s2 = re.sub(re.compile('(\《編者按\》本欄搜羅即市傳聞\，惟消息未經證實\，\《經濟通\》亦不保證內容之準確性。\.+$)'), '', s2)
    s2 = re.sub(re.compile('(\《編者按\》本欄搜羅即市傳聞\，惟消息未經證實\，\《經濟通\》亦不保證內容之準確性\。)'), '', s2)
    s2 = re.sub(re.compile('(\《編者按\》本欄搜羅即市傳聞\，惟消息未經證實\，\《經濟通\》亦不保證內容之準確)'), '', s2)

    s2 = re.sub(re.compile('(\(筆者為證監會持牌人士，無持有上述股份\))'), '', s2)
    s2 = re.sub(re.compile('(\*筆者為註冊持牌人士\，並無持有上述股份)'), '', s2)
    s2 = re.sub(re.compile('(\(筆者為註冊持牌人士\，並無持有上述股份\）)'), '', s2)

    s2 = re.sub(re.compile(
        '(\*編者按\: 本文只供參考之用\，並不構成要約\、招攬或邀請\、誘使\、任何不論種類或形式之申述或訂立任何建議及推薦\，讀者務請運用個人獨立思考能力自行作出投資決定\，如因相關建議招致損失\，概與\《環富通\》\、編者及作者無涉\。)'),
                '', s2)
    s2 = re.sub(re.compile(
        '(\*編者按\: 本文只供參考之用\，並不構成要約\、招攬或邀請\、誘使\、任何不論種類或形式之申述或訂立任何建議及推薦\，讀者務請運用個人獨立思考能力自行作出投資決定\，如因相關建議招致損失\，概與\《經濟通\》\、編者及作者無涉\。)'),
                '', s2)

    s2 = re.sub(re.compile('(註\: 保險產品須受有關保單的條款及細則約束\。)'), '', s2)
    s2 = re.sub(re.compile('(\〈有關牌照詳情\，請閱以下連結\:$)'), '', s2)
    s2 = re.sub(re.compile('(\d{2,4}年\d{1,2}月\d{1,2}日$)'), '', s2)

    s2 = re.sub(re.compile('(\(本欄逢周\w+刊出\))'), '', s2)
    s2 = re.sub(re.compile('(\【etnet一App通天下\】獨家\「銀行匯率比較\」功能:匯市靚價一眼通，一按直達交易商，買入賣出好輕鬆!立即下載\>\>etnet財經\.生活AppiOS\:)'), '',
                s2)

    s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\)\w+$)'), '', s2)  ## ----remove stockcodes
    s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\))'), '', s2)

    s2 = re.sub(re.compile(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+'), '', s2)  ## ----remove emails
    s2 = re.sub(re.compile('(\(\w+\）$)'), '', s2)


    s2=re.sub('﹒','.',s2)
    s_list=DummySentencizer(s2, split_characters=['。」','。',';','!','*']).sentences
    return s_list



#------------------------------------------
#preprocess_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\{}'.format(today_str)
data_dir= r'sentences'

# news_type='Total'
# news_folder=os.path.join(master_dir, news_type)

Sentences_filename='sentences.txt'
Sentences_file=os.path.join(data_dir,Sentences_filename)
#
# newscontent_filename='newscontent.txt'
# newscontent_file=os.path.join(data_dir,newscontent_filename)

#
# topicID_filename='topcIDs.txt'
# topicID_file=os.path.join(news_folder,topicID_filename)
# topicIDs=tuple(int(id) for id in pd.read_csv(topicID_file,names=['topicID'])['topicID'].values)


#
sql_select_query=f"""select newsID,headline, content from {mysql_tablename} where newsDate >='2020-01-01'  and  newsDate <='2020-03-15' and relCategory not like "%F1%" and relCategory not like "%L8%"  and relCategory not like "%L1%" and relCategory not like "%L11%"  and relCategory not like "%M9%";"""
mysql_cursor.execute(sql_select_query)


for row in tqdm(mysql_cursor.fetchall()):
        newsID=row[0]
        headline=row[1]
        content=row[2]

        headline= text_preprocessing(headline)
        content = text_preprocessing(content)

        # processed_content= ''.join([s for s in headline+[' ']+content if s !='*' ])
        # with open(newscontent_file, 'a', encoding='utf-8') as f:
        #     f.write(processed_content+'\n')


   # # sentences= DummySentencizer(content, split_characters=['。」','。',';','!','*']).sentences
        Sentences= [headline] + content#sentences
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

        with open(Sentences_file, 'a', encoding='utf-8') as f:
            for ss in sentences_clean:
                if type(ss) == list:
                    for s in ss:
                        #print(len(s.replace('*','')),s.replace('*',''))
                        f.write(s.replace('*','')+'\n')

                   # sentences_file.write(s)
                    #sentences_file.write('\n')
                else:
                    #print(len(ss.replace('*','')), ss.replace('*',''))
                    f.write(ss.replace('*', '') + '\n')
        print('\n\n')
   #
   #



mysql_conn.close()
