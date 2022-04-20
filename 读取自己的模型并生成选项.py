import json
import transformers
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
import pandas as pd
import numpy as np
import torch
from transformers import BertModel,BertTokenizer
from transformers  import *
from torch.utils.data import TensorDataset, random_split
import warnings
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW
from torch.autograd import Variable
import time
from transformers import get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")

sigmoid = nn.Sigmoid()
model=torch.load("216.pth")
tokenizer = BertTokenizer.from_pretrained(r"C:\Users\86189\Desktop\1\bert-law\vocab.txt")
df=pd.read_excel("test0.xlsx")
sentences=df.text.values
num=df.num.values
def predict(logits):
    res = torch.argmax(logits, 1)
    return res
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m=['A','B','C','D']
while(True):
    print("请选择回答的题目")
    n=eval(input())
    s=[]
    for a in range(len(sentences)):
        if num[a]==n:
            s.append(sentences[a])
    l=[]
    a=[]
    if len(s)!=0:
        for i in s:
            encoded_dict = tokenizer.encode_plus(
                        i,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
            l.append(encoded_dict['input_ids'])
            a.append(encoded_dict['attention_mask'])
        x=torch.cat(l, dim=0)
        x=x.to(device)
        y=torch.cat(a, dim=0)
        y=y.to(device)
        output=model(x,attention_mask=y)
        logits = output.logits
        p=predict(sigmoid(logits))
        if 0 not in p:
            p=[]
            for i in range(4):
                p.append(sigmoid(logits)[i][0])
            i=p.index(max(p))
            print(m[i])
        else:
            pr=[]
            for i in range(4):
                if p[i]==0:
                    pr.append(m[i])
            print(pr)
