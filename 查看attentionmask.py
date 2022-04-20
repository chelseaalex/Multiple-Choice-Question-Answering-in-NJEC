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
warnings.filterwarnings("ignore")

path=r"C:\Users\86189\Desktop\1\bert-law"
#bert-base-chinese
#model = BertModel.from_pretrained(path)
tokenizer = BertTokenizer.from_pretrained(r"bert-law\vocab.txt")
torch.set_default_tensor_type(torch.FloatTensor)



x1=pd.read_excel("train0.xlsx")
x2=pd.read_excel("train1.xlsx")
x=pd.concat([x1, x2], axis=0, ignore_index=True)

print(len(x))
sentences=x.text.values
ans=x.ans.values
num=x.num.values
"""
max_len=0
for m in x['text']:
    if(m!=1):
        input_id=tokenizer.encode(m,add_special_tokens=True)
        max_len=max(max_len,len(input_id))
print(max_len)
"""
#max_len=838
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []


for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    # encoded_dict字典形式返回有三类 input_ids，attention_mask，token_type_ids
    # 分别加入到列表input_ids和attention_mask中
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

labels = torch.tensor(ans).float()




# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])
































