import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
import c_model 
import preprocess 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("GPU Index: ",torch.cuda.current_device())

###############
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator,SRC,TRG = preprocess.prcoess(BATCH_SIZE,device)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 1
CLIP = 1
###############

enc = c_model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = c_model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = c_model.Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('chatbot1-model.pt'))

post = ""

while (post != "quit"):
    post = input("User: ")
    model.eval() 

    tokenized = preprocess.tokenize_src(post) 
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