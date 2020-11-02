
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


nlp = spacy.load('en')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU Index: ",torch.cuda.current_device())

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

SRC = Field(tokenize = tokenize_src, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_tgt, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

from torchtext.data import TabularDataset

tv_datafields = [("src", SRC), 
                 ("trg", TRG)]

print("Pre-processing....")


train_data, valid_data , test_data= TabularDataset.splits(
               path="./data", # the root directory where the data lies
               train='train.csv', validation="valid.csv", test = "test.csv" ,
               format='csv',
               skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
               fields=tv_datafields)


SRC.build_vocab(train_data, min_freq = 1)
TRG.build_vocab(train_data, min_freq = 1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = c_model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = c_model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = c_model.Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('chatbot1-model.pt'))

post = ""

while (post != "quit"):
    post = input("User: ")
    model.eval() 

    tokenized = tokenize_src(post) 
    tokenized = ['<sos>'] + [t.lower() for t in tokenized] + ['<eos>']
    numericalized = [SRC.vocab.stoi[t] for t in tokenized]
    sentence_length = torch.LongTensor([len(numericalized)]).to(device)
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    
    with torch.no_grad():
        output_tensor_probs= model(tensor,tensor, 0) #pass through model to get response probabilities
        output_tensor = torch.argmax(output_tensor_probs.squeeze(1), 1) #get response from highest probabilities
        output = [TRG.vocab.itos[t] for t in output_tensor][1:] #ignore the first token, just like we do in the training loop
        #output = output[:output.index('eos')]
        if('<eos>' in output):
            #print('index:',output.index('<eos>'))
            output = output[:output.index('<eos>')]
        output =' '.join(output)
        print("")
   
    print("LazyBot: ",output)
    print("####")