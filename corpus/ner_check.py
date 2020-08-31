import pandas as pd
import numpy as np
import json
from datetime import date
import os


today_str=date.today().strftime('%Y-%m-%d').replace('-','')
#
# e_ner_folder=r'C:\Users\marcus\Desktop\etnet_2\preprocessing\ner_before_training'
# e_ner_filename='diva_finance.txt'
# e_ner_file=os.path.join(e_ner_folder,e_ner_filename)
# e_ner=pd.read_csv(e_ner_file,sep=' ',header=0,encoding='utf-8')['word'].values
#
#
NER_resources_folder= r'NER_resources'

ner_results_filename='ner_results.txt'
ner_results=pd.read_csv(os.path.join(NER_resources_folder, ner_results_filename), sep=' ', names=['word', 'ner'], encoding='utf-8')


print(ner_results.shape[0])
ner_results.drop_duplicates('word', inplace=True)
print(ner_results.shape[0])
print(ner_results.ner.unique())
ner_emtpy=ner_results[ner_results.isna().any(axis=1)]
ner_results=ner_results[~(ner_results.isna().any(axis=1))]
print(ner_results.shape[0])
print(ner_emtpy.shape[0])
print(ner_emtpy)
print(ner_results[ner_results['ner']=='TERN'])
ner_emtpy.to_csv('ner_emtpy.txt',sep=' ',header=0,index=None,encoding='utf-8')
ner_results.to_csv(os.path.join(NER_resources_folder, ner_results_filename), sep=' ', header=0, index=None, encoding='utf-8')




with open(os.path.join(NER_resources_folder, 'ner2pos.json'), 'r') as fp:
    ner2pos=json.load(fp)

jieba_dict_df=ner_results.copy()
jieba_dict_df['freq']=5
jieba_dict_df['pos']=jieba_dict_df['ner'].apply(lambda x: str(ner2pos[str(x)]))
jieba_dict_df=jieba_dict_df[['word','freq','pos']]
print(jieba_dict_df.head())

jieba_dict_df.to_csv(os.path.join(NER_resources_folder, 'userdict_ner.txt'), sep=' ', header=None, index=None, encoding='utf-8')

