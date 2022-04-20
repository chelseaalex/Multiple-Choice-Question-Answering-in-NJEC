import json
import pandas as pd
import numpy as np
def read_json(data_path):
    a=[]
    with open(data_path, 'r',encoding = 'utf-8') as f:
        # content = json.load(f) #报错x
        a=[]
        content = f.readlines() 
        for line in content:
            a.append(json.loads(line))
    return a

data1=read_json("0_train.json")


data3=read_json("0_test.json")
data2=read_json("1_test.json")
data4=read_json("1_train.json")

df=pd.DataFrame(columns=['text','ans','num'])
for i in range(len(data1)):
    m=data1[i]
    if m['answer']==[]:
        continue
    else:
        k=0
        for an in  m['answer']:
            if an in ['A','B','C','D']:       
                s=pd.Series({'text':m["statement"]+m['option_list'][an],'ans':0,'num':i})
                df=df.append(s,ignore_index=True)
        for a in ['A','B','C','D']:
            if a not in m['answer']:
                s=pd.Series({'text':m["statement"]+m['option_list'][a],'ans':1,'num':i})
                df=df.append(s,ignore_index=True)
df.to_excel("train0.xlsx")
"""
df2=pd.DataFrame(columns=['text','ans','num'])
for i in range(len(data4)):
    m=data4[i]
    if m['answer']==[]:
        continue
    else:
        k=0
        for an in  m['answer']:
            if an in ['A','B','C','D']:       
                s=pd.Series({'text':m["statement"]+m['option_list'][an],'ans':0,'num':i+len(data1)})
                df2=df2.append(s,ignore_index=True)
        for a in ['A','B','C','D']:
            if a not in m['answer']:
                s=pd.Series({'text':m["statement"]+m['option_list'][a],'ans':1,'num':i+len(data1)})
                df2=df2.append(s,ignore_index=True)

df2.to_excel("train1.xlsx")

df3=pd.DataFrame(columns=['text','num'])
for i in range(len(data2)):
    m=data2[i]
    for an in  ['A','B','C','D']:       
        s=pd.Series({'text':m["statement"]+m['option_list'][an],'num':i})
        df3=df3.append(s,ignore_index=True)
df3.to_excel("test0.xlsx")

df4=pd.DataFrame(columns=['text','num'])
for i in range(len(data3)):
    m=data2[i]
    for an in  ['A','B','C','D']:       
        s=pd.Series({'text':m["statement"]+m['option_list'][an],'num':i})
        df4=df4.append(s,ignore_index=True)
df4.to_excel("test1.xlsx")
"""
