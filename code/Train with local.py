# -*- coding: utf-8 -*-
"""Preprocessing_without_missing_info_bart.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aywqXHNGz-XOO5P3lJ2bDZpWSSkEKL6A
"""

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
import zipfile
import os
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def flatten(data,keys_list):

    dict1 = {}
    temp={}
    dict1["question_Schema"]=[]
    dict1['output_questions']=[]
    dict1["description_schema"]=[]
    dict1["title"]=[]
    dict1["category"]=[]
    temp["question_Schema"]=[]
    temp['output_questions']=[]
    
    
    count=0
    for key in keys_list:
        dict1["description_schema"].append(data['description_schema'][key])  
        dict1["title"].append(data['title'][key] )
        dict1["category"].append(data['category'][key])
        for j in range(len(data['questions'][key])):
   
            temp["question_Schema"].append(data['questions'][key][j]['schema'])
            temp['output_questions'].append(data['questions'][key][j]['question'])
        dict1['question_Schema'].append(temp['question_Schema'][count:count+len(data['questions'][key])+count])
        dict1['output_questions'].append(temp['output_questions'][count:count+len(data['questions'][key])+count])
        count=len(data['questions'][key])+count
      

    return dict1
  
def add_sep_Q_Schema(out):
    for i in range(len(out['question_Schema'])):
        temp=[]
        for j in range(len(out['question_Schema'][i])):
            for k in range(len(out['question_Schema'][i][j])):
                #print(out['question_Schema'][i][j][k])
                temp.append(out['question_Schema'][i][j][k])
                temp.append("</s>")
        #print(temp)
        out['question_Schema'][i]=temp
    return out
  
def replace_str(word):
  w1 = word.replace('[','')
  w2 = w1.replace(']','')
  return w2

def listToString(s): 
    
    # initialize an empty string
    str1 = "</s>" 
    
    # return string  
    return (str1.join(s))

def add_sep(s):
  
  str1 = "</s>" 
  return str1+s

def batch_processing_tokenizer (batch, tokenizer,max_source_length, max_target_length):

  question_s,label ,tit_cat_desc = batch['question_Schema'], batch['labels'], batch['title_category_desc']

  source_tokenized_dataset = tokenizer(question_s ,tit_cat_desc,
                             padding="max_length", truncation=True, max_length=max_source_length)   # Does Bart tokenizer tokenize only first two columns

  target_tokenized_dataset = tokenizer(label ,padding="max_length", truncation=True, max_length=max_target_length)

  batch = {k: v for k, v in source_tokenized_dataset.items()}

  # Ignore padding in the loss
  batch["labels"] = [
      [-100 if token == tokenizer.pad_token_id else token for token in l]
      for l in target_tokenized_dataset["input_ids"]
  ]
  return batch



_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "facebook/bart-base"
_TOKENIZER_FOR_DOC = "facebook/bart-base"

model = BartForConditionalGeneration.from_pretrained(_CHECKPOINT_FOR_DOC)
config = BartConfig.from_pretrained(_CONFIG_FOR_DOC)
tokenizer = BartTokenizer.from_pretrained(_TOKENIZER_FOR_DOC, use_fast=True)    

if torch.cuda.is_available():
    model.cuda()


with zipfile.ZipFile('/content/drive/MyDrive/Missing_schema/schema.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/Extract_files')

files= getListOfFiles('/content/Extract_files')
Cons_X = list()
cons_y = list()

files_xls = [f for f in files if f[-5:] == '.json']
print(files_xls)
df = pd.DataFrame()
df_new =pd.DataFrame()
for f in files_xls:
  df=pd.read_json(f)
  df=df.T
  df_new=pd.concat([df_new, df])


ind= df_new.index.values

    
to_dic = df_new.to_dict()
out=flatten(to_dic,ind)

jk = pd.DataFrame.from_dict(out)

jk=add_sep_Q_Schema(jk)
jk['question_Schema'] = jk['question_Schema'].astype(str)
jk['question_Schema'] = jk['question_Schema'].apply(replace_str)

jk['labels'] = jk['output_questions'].apply(listToString)
jk['description_schema'] = jk['description_schema'].apply(listToString)
jk['category'] = jk['category'].apply(add_sep)
jk['title_category_desc'] = jk['title']+jk['category']+jk['description_schema']

jk = jk.drop(['category','title','output_questions','description_schema'],axis=1)

from datasets import Dataset
dataset = Dataset.from_pandas(jk)
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()
encoder_max_length = 512
decoder_max_length = 128

train_data = train_data_txt.map(lambda batch: batch_processing_tokenizer(
        batch, tokenizer,encoder_max_length, decoder_max_length
    ),
    batched=True, 
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_processing_tokenizer(
        batch, tokenizer,encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=50,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=4,  # demo
    per_device_eval_batch_size=2,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
)

trainer.train()
model.generate()
val_data = validation_data_txt.select(range(30))
val_data_test =tokenizer(val_data['question_Schema'],val_data['title_category_desc'],
                              truncation=True,return_tensors="pt",padding=True)

input_ids = val_data_test.input_ids.to(model.device)
attention_mask = val_data_test.attention_mask.to(model.device)
outputs = model.generate(input_ids, attention_mask=attention_mask,num_beams = 5,max_length = 200 )
output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_str