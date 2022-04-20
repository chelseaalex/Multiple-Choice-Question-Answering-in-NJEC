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
    #返回有三类 input_ids，attention_mask，token_type_ids    
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


dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 8
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially. 验证集不用管顺序
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = RandomSampler(val_dataset), # Pull out batches sequentially.SequentialSampler
            batch_size = batch_size 
        )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW

#model = BertForSequenceClassification.from_pretrained(r"C:\Users\86189\Desktop\1\bert-law",num_labels = 2,output_attentions = False,output_hidden_states = False)
model=torch.load("215.pth")
model.to(device)

from torch import optim


criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

#########
epoch = 10
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5,
                     eps = 1e-8
                     )
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def predict(logits):
    res = torch.argmax(logits, 1)
    return res

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, axis=1).flatten()    # 取出最大值对应的索引
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)

import datetime
 
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.    四舍五入
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, axis=1).flatten()    # 取出最大值对应的索引
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)



from torch.autograd import Variable
import time

pre = time.time()

accumulation_steps = 8

training_stats=[]
for i in range(epoch):
    model.train()
    total_train_loss = 0
    t0=time.time()
    for step, batch in enumerate(train_dataloader):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].long().to(device)
        #b_labels=Variable(b_labels,requires_grad=True)
        model.zero_grad()
        
        output = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask,
                             labels=b_labels)
        logits = output.logits
        #loss = criterion(predict(sigmoid(logits)).to(torch.float32), b_labels.to(torch.float32))
        loss=output[0]
        total_train_loss += loss.item()
        #loss = Variable(loss,requires_grad=True)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if ((step+1) % 200) == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                i+1, step, len(train_dataloader), 100. *
                step/len(train_dataloader), loss.item()
            ))

        if step == len(train_dataloader)-2:
            # 在每个 Epoch 的最后输出一下结果
            pred = flat_accuracy(logits, b_labels)
            
            print('labels:', b_labels)
            print('pred:', pred)
            print('pred_label',predict(sigmoid(logits)).to(torch.float32))
            
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    
    # 打印结果
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    
    # ========================================
    #               Validation
    # ========================================
    # 在验证集查看
 
    print("")
    print("Running Validation...")
 
    t0 = time.time()
 
    # 将模型置于评估模式 不使用BatchNormalization()和Dropout()
    model.eval()
    """
    # 跟踪变量
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    nb_eval_examples = 0
 
    # 在每个epoch上评估
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids, b_input_mask, b_labels = batch
        
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        
        output = model(b_input_ids.long(), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                       labels=b_labels.long()
                       )
            
        # 计算验证损失
        logits = output.logits
        loss=output[0]
        #loss = criterion(predict(sigmoid(logits)).to(torch.float32), b_labels.to(torch.float32))
        total_eval_loss += loss.item()
 
        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')
 
        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        total_eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
 
    # 返回验证结果
    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    # 计算平均复杂度
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # 时间
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
 
    # 记录这个epoch的所有统计数据。 方便后面可视化
    training_stats.append(
        {
            'epoch': i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )"""
    #torch.save(model,"216.pth")
    
 
print("")
print("Training complete!")
 
#print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))































