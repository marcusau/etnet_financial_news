import numpy as np
import pandas as pd
import re
from numpy import log, min
import os

from datetime import datetime, timedelta,date
from collections import defaultdict


import jieba_fast as jieba

#import openCC
from tqdm import tqdm
import time,json
import pytz
import swifter





############# import jieba dictionary #####################################################################################################################################################################

#
textdata_dir= r'NER_resources'
userdict_old_filename='userdict_ner.txt'
temp_filename= 'temp.txt'

userdict_old_file=os.path.join(textdata_dir, userdict_old_filename)
userdict_new_file=os.path.join(textdata_dir, temp_filename)

userdict_df=pd.read_csv(userdict_old_file,names=['word','freq','pos_tag'],delimiter=' ',encoding='utf-8')[['word']]



################# load punctuations ###############################################################################################################################################################################################################################



def cal_jieba_freq(text):
    try:
        jieba_freq=max(jieba.suggest_freq(text), len(text)**3)
    except:
        print(text)
        jieba_freq=3
    return jieba_freq


def update_userdict(userdict_old_file,userdict_new_file):

    userdict_old_df=pd.read_csv(userdict_old_file,names=['word','freq','pos_tag'],sep=' ',encoding='utf-8')#[['word',]]
    userdict_old_df['word']=userdict_old_df['word'].astype('str')
   # pool = Pool(processes=num_of_cpu)
    #userdict_old_df =pool.map(read_userdict,userdict_old_file)
    #pool.close()
    #userdict_old_df.drop('word_len', axis=1, inplace=True)
    userdict_old_df['word_len']=userdict_old_df['word'].str.len()
    userdict_old_df.sort_values('word_len',ascending=True,inplace=True)
    #for i in tqdm(range(1,37)):
    word_len_list=sorted(userdict_old_df['word_len'].value_counts().index.tolist())
    for i in tqdm(word_len_list):
        try:
            jieba.load_userdict(userdict_new_file)
            print('loaded new userdict')
        except:
            print('cannot loaded new userdict')
            pass

        df_processing = userdict_old_df[(userdict_old_df['word_len'] == i)]
        print('words_len: {}, no. of words in old userdict:{}'.format(i, df_processing.shape[0]))
        #if df_processing.shape[0] == 0:
         #   print('No word with length of {}'.format(i))
          #  pass
        #else:
       # print('processing word with length of {}'.format(i))
        df_processing.drop('word_len', axis=1, inplace=True)
        df_processing.drop_duplicates('word', keep='last', inplace=True)
        df_processing['freq'] = df_processing['word'].swifter.apply(cal_jieba_freq)
        print('words_len: {},  new userdict:{}'.format(i, df_processing.shape[0]))
        #print(df_processing.head(20))
        df_processing.to_csv(userdict_new_file, mode='a', sep=' ', index=None, header=None, encoding='utf-8')


######################### Run Dictionary Cleaning   ###################################################################################################
def run_userdict_clean():
    start_time=time.time()
    update_userdict(userdict_old_file,userdict_new_file)

    userdict_new_df = pd.read_csv(userdict_new_file, names=['word', 'freq','pos'], sep=' ', encoding='utf-8')
    #pool = Pool(processes=num_of_cpu)
    #userdict_new_df = pool.map(read_userdict, userdict_new_file)
    userdict_new_df.drop_duplicates('word',keep='last',inplace=True)
    #userdict_new_df.dropna(axis=0,inplace=True)
    userdict_new_df.to_csv(userdict_old_file, mode='w', sep=' ', index=None, header=None, encoding='utf-8')
    os.remove(userdict_new_file)
    finish_time=time.time()
    print('processing time:',(finish_time-start_time),' seconds')


run_userdict_clean()
#