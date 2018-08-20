# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:22:43 2018

@author: JG
"""

import os
import sys
import logging
import pandas as pd
import pandas.io.sql as psql


try:
    import psycopg2 as pg
    import psycopg2.extras
except:
    print( "Install psycopg2")
    exit(123)
    
PG_CONN_STRING = "dbname='postgres' port='5432' user='postgres' password='phludphlud'"
dbconn = pg.connect(PG_CONN_STRING)
cursor = dbconn.cursor()

row_count = int(pd.read_sql('SELECT COUNT(*) from review_view', con=dbconn).values)

print(row_count)
chunksize = 50000
review_df = pd.DataFrame(columns = ['text', 'stars'])

# Load review into Pandas DataFrame
for i in range(int(row_count/chunksize) +1):
    query = 'SELECT text, stars from review_view LIMIT {chunksize} OFFSET {offset}'.format(offset = i*chunksize, chunksize=chunksize)
    review_df = review_df.append(pd.read_sql_query(query, con=dbconn))
    print("{} rows have been loaded to dataframe.".format(i*chunksize))
    
#%%

print(review_df.head()) 
print(review_df.shape)

#%%
review_df['stars'] = review_df['stars'].astype(int)
print(review_df.dtypes)
filtered_df = review_df[review_df['stars'].isin([1,5])]
print(filtered_df.dtypes)
print(filtered_df.shape)

#%%
# Convert labels to 0-1 scale (1 star becomes 0, 5 star becomes 1)
def convert_scale(x):    
    if x == 1: x = 0
    if x == 5: x = 1
    return x

filtered_df['stars'] = filtered_df['stars'].apply(convert_scale)
print(filtered_df.head())

#%%
filtered_df.to_csv("filtered.csv")

#%%
# We will now use PyTorch to build a simple RNN which can then be trained on the data
import torch
from torchtext import data, datasets

SEED = 1337

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Based on bi-grams concept from "FastText" model (Joulin et al., 2016)
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
LABEL = data.LabelField(tensor_type=torch.FloatTensor)


#%%
datafields = [("text", TEXT), ("stars", LABEL)]
train = data.TabularDataset(
        path = 'filtered.csv', format='csv', skip_header=True, fields=datafields) 


#%%
# Split training data into train, valid and test sets
train, train_dummies = train.split(split_ratio=[0.1, 0.9]) # Only use 10% of the data, otherwise training takes forever...
train, valid, test = train.split(split_ratio=[0.98, 0.01, 0.01])

print('Train length:', len(train))
print('Valid length:', len(valid))
print('Test length:', len(test))

TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train)

print("Vocab built.")

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.text), 
    repeat=False)

print("Iterators generated.")
#%%
# Build the neural network based on FastText paper
import torch.nn as nn

class FastText(nn.Module):
    def __init__(self, vocab, embedding_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self,x):
        embedded = self.embedding(x)
        
        embedded = embedded.permute(1,0,2)
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        
        return self.fc(pooled)

print("Input dimensions:", len(TEXT.vocab))
INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
OUTPUT_DIM = 1

#%%
# Build the NN model
model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

#%%
# Now we are ready to train the model!
import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

device = torch.device('cpu')

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    
    accuracy = correct.sum()/len(correct)
    
    return accuracy

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.stars)
        
        acc = binary_accuracy(predictions, batch.stars)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.stars)
            
            acc = binary_accuracy(predictions, batch.stars)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#%%
# Train for a number of epochs
N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}%, Val. Loss: {:.3f}, Val. Acc: {:.2f}%'.format(epoch + 1, train_loss, train_acc * 100, valid_loss, valid_acc * 100))

#%%
# Just for fun, let's test on some custom user inputs
import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = F.sigmoid(model(tensor))
    return prediction.item()

print(predict_sentiment("This restaurant was amazing."))

print(predict_sentiment("The food was awful."))

#%%
for i in train_iterator:
    print(i)