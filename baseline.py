import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
from torchtext.data import TabularDataset
from nltk.translate.bleu_score import sentence_bleu


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

nlp = spacy.load('en')

def tokenize_en(text):
 
    return [tok.text for tok in nlp.tokenizer(text) if not tok.is_punct]



SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)


tv_datafields = [("src", SRC), 
                ("trg", SRC)]


train_data, valid_data , test_data= TabularDataset.splits(
            path="/content/drive/My Drive/", # the root directory where the data lies
            #train='train_1.csv', validation="train_1.csv", test = "train_1.csv" ,
            train='train.csv', validation="valid.csv", test = "test.csv" ,
            format='csv',
            skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=tv_datafields)


print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))
print(vars(valid_data.examples[0]))
print(vars(test_data.examples[0]))
