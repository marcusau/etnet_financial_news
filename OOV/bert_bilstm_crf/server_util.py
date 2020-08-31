
import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from tqdm import tqdm
import re
from full_width_to_half_width import full_width_to_half_width





def text_preprocessing(sentences):
        sentences=re.sub('\n|\s', '', sentences)
        sentences = sentences.replace('\u3000', '').replace('\ufeff', '').replace('\xa0', '')
        s1=DummySentencizer(sentences,split_characters=['，','。','*',';','。」','。',';','!','，','、']).sentences
        s2=''.join([full_width_to_half_width(s[:-1])+s[-1] for s in s1])
        s2 = s2.replace('\u3000', '').replace('\ufeff', '').replace('\xa0', '')
        s2 = re.sub('﹒', '.', s2)
        s2 = re.sub(re.compile('https?://\S+'), '', s2)                                                               ## ----remove html
        s2 = re.sub(re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), '',s2)  ## ----remove html
        s2 = re.sub(re.compile('\(\d{3,6}\)'),'',s2)
        s2 = re.sub(re.compile('(\(滬:\d+\))'),'',s2)
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

        s2 = re.sub(re.compile('(\*編者按\: 本文只供參考之用\，並不構成要約\、招攬或邀請\、誘使\、任何不論種類或形式之申述或訂立任何建議及推薦\，讀者務請運用個人獨立思考能力自行作出投資決定\，如因相關建議招致損失\，概與\《環富通\》\、編者及作者無涉\。)'), '', s2)
        s2 = re.sub(re.compile('(\*編者按\: 本文只供參考之用\，並不構成要約\、招攬或邀請\、誘使\、任何不論種類或形式之申述或訂立任何建議及推薦\，讀者務請運用個人獨立思考能力自行作出投資決定\，如因相關建議招致損失\，概與\《經濟通\》\、編者及作者無涉\。)'), '', s2)
        s2 = re.sub(re.compile( '(\*編者按: 本文只供參考之用，並不構成要約、招攬或邀請、誘使、任何不論種類或形式之申述或訂立任何建議及推薦，讀者務請運用個人獨立思考能力自行作出投資決定，如因相關建議招致損失，概與《經濟通》、編者及作者無涉。)'),  '', s2)



        s2 = re.sub(re.compile('(註\: 保險產品須受有關保單的條款及細則約束\。)'), '', s2)
        s2 = re.sub(re.compile('(\〈有關牌照詳情\，請閱以下連結\:$)'), '', s2)
        s2 = re.sub(re.compile('(\d{2,4}年\d{1,2}月\d{1,2}日$)'), '', s2)



        s2 = re.sub(re.compile('(\(本欄逢周\w+刊出\))'), '', s2)
        s2 = re.sub(re.compile('(\【etnet一App通天下\】獨家\「銀行匯率比較\」功能:匯市靚價一眼通，一按直達交易商，買入賣出好輕鬆!立即下載\>\>etnet財經\.生活AppiOS\:)'), '', s2)

        s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\)\w+$)'), '', s2)## ----remove stockcodes
        s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\))'), '', s2)

        s2 = re.sub(re.compile(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+'),'',s2)                                                 ## ----remove emails
        s2 = re.sub(re.compile('(\(\w+\）$)'), '', s2)
        s_list = DummySentencizer(s2, split_characters=['。」', '。', ';', '!', '*']).sentences
        return s_list

##----- text pre-processing --------------
delimiters=re.compile('，|。|\*')

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