#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pandas as pd

from tqdm import tqdm, trange
import datetime
import time

import sqlalchemy
import mysql.connector

#####################################################################################################################################

mysql_host = '10.1.8.19'
mysql_port=3306
mysql_user = 'root'
mysql_password = 'abcd1234'
mysql_schema= 'etnet_financial_news'
#ysql_tablename='news_test1'

mysql_conn = mysql.connector.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_schema,  auth_plugin='mysql_native_password')
mysql_cursor = mysql_conn.cursor()


#####################################################################################################################################

sqlalchemy_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_schema}')
sqlalchemy_connection = sqlalchemy_engine.connect()

#####################################################################################################################################
mysql_tablename1='test1_autotags'
sql_select_query1=f"""select autotag_ID,title from {mysql_tablename1};"""
mysql_cursor.execute(sql_select_query1)
autotag_records={autotag:autotag_ID for (autotag_ID,autotag) in tqdm(mysql_cursor.fetchall()) }


mysql_tablename2='news_test1'
sql_select_query=f"""select newsID,autotags from {mysql_tablename2} where newsDate >='2020-03-15' and relCategory not like "%F1%" and relCategory not like "%L8%"  and relCategory not like "%L1%" and relCategory not like "%L11%"  and relCategory not like "%M9%";"""
mysql_cursor.execute(sql_select_query)
newsid_records=[(newsID, autotags.split('\t')) for (newsID,autotags) in tqdm(mysql_cursor.fetchall())  ]


tag_article_match=[]
for (newsID, autotags_list) in tqdm(newsid_records):
    for autotag in autotags_list:
        tag_article_match.append((autotag_records[autotag],newsID))


insert_autotags_query = "insert into test1_autotag_match (tid,newsID)  VALUES ( %s, %s );"
mysql_cursor.executemany(insert_autotags_query, tag_article_match)
mysql_conn.commit()


mysql_conn.close()



