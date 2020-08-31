#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import re
from datetime import date
import sys
import datetime
import time
import pandas as pd
import codecs
import numpy as np

from tqdm import tqdm, trange

data_dir=r'C:\Users\marcus\PycharmProjects\KG_test\OOV\bert_bilstm_crf\output'
data_filename='bert_bilstm_crf.txt'
data_file=os.path.join(data_dir,data_filename)
data_df=pd.read_csv(data_file,sep=' ',names=['word','ner'],encoding='utf-8')
print(data_df.shape[0])
data_df.drop_duplicates('word',inplace=True)
print(data_df.shape[0])
data_df.to_csv(data_file,sep=' ',index=None,header=0,encoding='utf-8')
