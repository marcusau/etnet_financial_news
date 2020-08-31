import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from tqdm import tqdm
import re
from full_width_to_half_width import full_width_to_half_width

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens


class NerProcessor(object):
    def read_data(self, input_file):  #### """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:

            lines, words, labels = [], [], []

            for line in tqdm(f.readlines()):
                contends = line.strip().strip('\n')
                tokens = line.strip().strip('\n').split(" ")
                try:
                    if len(tokens) == 2:
                        words.append(tokens[0].strip())
                        labels.append(tokens[1].strip())
                    else:
                        if len(contends) == 0 and len(words) > 0:
                            label, word = [], []
                            for l, w in zip(labels, words):
                                if len(l) > 0 and len(w) > 0:
                                    label.append(l)
                                    word.append(w)
                            lines.append([' '.join(label), ' '.join(word)])
                            words, labels = [], []
                except:
                    print('Cannot process in read_data in NERProcessor of util.py', line)

            return lines

    def get_labels(self, args):
        labels = set()
        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
            print(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)

        else:
            # get labels from train data
            print(f"loading labels info from train file and dump in {args.output_dir}")
            with open(args.train_file) as f:
                for line in f.readlines():
                    tokens = line.strip().strip('\n').split(" ")

                    if len(tokens) == 2:
                        labels.add(tokens[1])

            if len(labels) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump(labels, f)
            else:
                print("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}

        return labels

    def get_examples(self, input_file):

        examples = []

        lines = self.read_data(input_file)

        for i, line in tqdm(enumerate(lines)):
            guid, label, text = str(i), line[0], line[1]

            examples.append(InputExample(guid=guid, text=text, label=label))

        return examples


def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     print"Writing example %d of %d" % (ex_index, len(examples)))

        textlist = example.text.split(" ")
        labellist = example.label.split(" ")
        assert len(textlist) == len(labellist)

        tokens, labels, ori_tokens = [], [], []

        for i, word in enumerate(textlist):
            # 防止wordPiece情况出现，不过貌似不会

            token = tokenizer.tokenize(word)

            tokens.extend(token)
            label_1 = labellist[i]
            ori_tokens.append(word)

            # 单个字符不会出现wordPiece
            for m in range(len(token)):
                if m == 0:

                    labels.append(label_1)
                else:
                    print(token, 'token is longer than 1')
                    if label_1 == "O":
                        labels.append("O")
                    else:
                        labels.append("I")

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]

        ntokens, segment_ids, label_ids = [], [], []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")

        segment_ids.append(0)
        label_ids.append(label_map["O"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)

        # assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, ori_tokens: {ori_tokens}, ntokens:{ntokens}"

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in ntokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        features.append(  InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids,  ori_tokens=ori_tokens))

    return features

def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_examples(filepath)
    label_list = args.label_list

    features = convert_examples_to_features(  args, examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return examples, features, data



def text_preprocessing(sentences):

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



        s2 = re.sub(re.compile('(註\: 保險產品須受有關保單的條款及細則約束\。)'), '', s2)
        s2 = re.sub(re.compile('(\〈有關牌照詳情\，請閱以下連結\:$)'), '', s2)
        s2 = re.sub(re.compile('(\d{2,4}年\d{1,2}月\d{1,2}日$)'), '', s2)



        s2 = re.sub(re.compile('(\(本欄逢周\w+刊出\))'), '', s2)
        s2 = re.sub(re.compile('(\【etnet一App通天下\】獨家\「銀行匯率比較\」功能:匯市靚價一眼通，一按直達交易商，買入賣出好輕鬆!立即下載\>\>etnet財經\.生活AppiOS\:)'), '', s2)

        s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\)\w+$)'), '', s2)## ----remove stockcodes
        s2 = re.sub(re.compile('(\(本文只屬作者個人意見\，並不代表作者所屬任何機構之立場\))'), '', s2)

        s2 = re.sub(re.compile(r'[\w\.-]+@[\w\.-]+(?:\.[\w]+)+'),'',s2)                                                 ## ----remove emails
        s2 = re.sub(re.compile('(\(\w+\）$)'), '', s2)
        return s2

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