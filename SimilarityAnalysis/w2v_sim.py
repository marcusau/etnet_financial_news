#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import random
import re

import numpy as np
import pandas as pd

from tqdm import tqdm, trange
import datetime
import time

import gensim
from sklearn.metrics.pairwise import cosine_similarity

import sqlalchemy
from sqlalchemy import engine, VARCHAR
import mysql.connector

#####################################################################################################################################

w2v_model_dir='pretrain_w2v'
w2v_model_filename='etnet_w2v.model'
w2v_model_file=os.path.join(w2v_model_dir,w2v_model_filename)
w2v_model = gensim.models.Word2Vec.load(w2v_model_file)
w2v_vocab=w2v_model.wv.vocab

#####################################################################################################################################

mysql_host = '10.1.8.19'
mysql_port=3306
mysql_user = 'root'
mysql_password = 'abcd1234'
mysql_schema= 'etnet_financial_news'
mysql_tablename='news_test1'

mysql_conn = mysql.connector.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_schema,  auth_plugin='mysql_native_password')
mysql_cursor = mysql_conn.cursor()


#####################################################################################################################################

sqlalchemy_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_schema}')
sqlalchemy_connection = sqlalchemy_engine.connect()

sql_select_query=f"""select newsID,autotags from {mysql_tablename} where newsDate >='2020-03-15' and relCategory not like "%F1%" and relCategory not like "%L8%"  and relCategory not like "%L1%" and relCategory not like "%L11%"  and relCategory not like "%M9%";"""
mysql_cursor.execute(sql_select_query)
records=mysql_cursor.fetchall()

autotags_w2vs=[]
for (newsID,autotags) in tqdm(records):

    autotags_w2v = [w for w in autotags.split('\t') if w in w2v_vocab ]
    if len(autotags_w2v )>0:
        autotags_w2vs.append((newsID,autotags_w2v))
    else:
        print(newsID,autotags)



for (newsID1, autotags_w2v1) in tqdm(autotags_w2vs):
    sims =[]
    #print(newsID1)

    # sql_select_query1 = f"""select autotags from {mysql_tablename} where newsID={newsID1};"""
    # mysql_cursor.execute(sql_select_query1)
    # print(mysql_cursor.fetchone()[0].split('\t'))
    for (newsID2, autotags_w2v2) in autotags_w2vs:
        if newsID1 !=newsID2:
           #
            sim=np.float64(w2v_model.n_similarity(autotags_w2v1, autotags_w2v2)).item()
            sims.append((newsID1, newsID2, sim))
           # print(type(sim),sim)
    sims.sort(key=lambda x: x[2],reverse=True)
    sims=sims[:20]
    #print(sims)


    insert_autotags_query = "insert into test1_articles_match (newsID1,newsID2,similarity)  VALUES ( %s, %s , %s);"
    mysql_cursor.executemany(insert_autotags_query,sims)
    mysql_conn.commit()

    #print(newsID1)
    # sim_ids=tuple([i[0] for i in sims[:20]])
    # #print(sims[:10])
    # sql_select_query_sim = f"""select newsID, autotags from {mysql_tablename} where newsID in {sim_ids};"""
    # mysql_cursor.executemany(sql_select_query_sim)
    # for (newsid_sim,autotags_sim) in mysql_cursor.fetchall():
    #     print(newsid_sim,autotags_sim)





    # autotags_w2v = [w for w in autotags if w in w2v_vocab ]
    # if len(autotags_w2v )>0:
    #     avg_w2v =np.mean(w2v_model[autotags_w2v], axis=0)
    #
    #     insert_autotags_query = "UPDATE {} SET autotags_w2v = '{}' WHERE newsID = {};".format(mysql_tablename,   avg_w2v, newsID)
    #     mysql_cursor.execute(insert_autotags_query)
    #     mysql_conn.commit()
    # else:
    #     print(newsID,autotags)


mysql_conn.close()

