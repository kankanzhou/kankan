
import os
import spacy

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from torchtext.data import Field
from torchtext.data import Field, BucketIterator
import sacrebleu
import math
import c_model
from torchtext.data import TabularDataset


nlp = spacy.load('en')

def tokenize_tgt(text):
    """
    Tokenizes post text from a string into a list of strings (tokens) 
    """
    return [tok.text for tok in nlp.tokenizer(text)]

def tokenize_src(text):
    """
    Tokenizes response text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp.tokenizer(text)]

def prcoess(BATCH_SIZE , device):


    SRC = Field(tokenize = tokenize_src, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_tgt, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    tv_datafields = [("src", SRC), 
                    ("trg", TRG)]


    train_data, valid_data , test_data= TabularDataset.splits(
                path="./data", # the root directory where the data lies
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


    SRC.build_vocab(train_data, min_freq = 1)
    TRG.build_vocab(train_data, min_freq = 1)


    print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        sort_key=lambda x: len(x.src), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        batch_size = BATCH_SIZE, 
        device = device)

    return train_iterator, valid_iterator, test_iterator ,SRC,TRG