
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

#######
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator,SRC,TRG = preprocess.prcoess(BATCH_SIZE,device)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10
CLIP = 1
#######

enc = c_model.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = c_model.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = c_model.Seq2Seq(enc, dec, device).to(device)
model.apply(c_model.init_weights)

optimizer = optim.Adam(model.parameters())
PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


best_valid_loss = float('inf')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = c_model.train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = c_model.evaluate(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'chatbot1-model.pt')
        
    end_time = time.time()
    epoch_mins, epoch_secs = c_model.epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('chatbot1-model.pt'))

test_loss = c_model.evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')