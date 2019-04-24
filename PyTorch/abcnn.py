import os
import copy
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

import math

import collections
from collections import Counter

from sklearn.utils import shuffle

import sys
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(text):
  text = text.translate(str.maketrans('', '', string.punctuation))
  words = text.split()
  return " ".join(word.lower() for word in words)

data = []

poscount = 0
negcount = 0

with open('../Datasets/rt-polarity.pos','r',encoding='latin1') as f:
  for line in f.readlines():
    data.append(preprocess(line[:-1]))
    poscount+=1
    
with open('../Datasets/rt-polarity.neg','r',encoding='latin1') as f:
  for line in f.readlines():
    data.append(preprocess(line[:-1]))
    negcount+=1

labels = np.zeros(poscount+negcount)
labels[:poscount] = 1

data,labels = shuffle(data,labels,random_state=0)

traincorpus,testcorpus,trainlabels,testlabels = train_test_split(data,labels,test_size=0.2,random_state=0)

traincorpus,valcorpus,trainlabels,vallabels = train_test_split(traincorpus,trainlabels,test_size=0.1,random_state=0)

ytrain = torch.from_numpy(trainlabels)
yval = torch.from_numpy(vallabels)
ytest = torch.from_numpy(testlabels)

trainlen = len(traincorpus)
vallen = len(valcorpus)
testlen = len(testcorpus)

words = []
for sentence in traincorpus:
  words+=sentence.split()

counter = Counter(words).most_common()
vocabulary = {}
vocabulary['<PAD>'] = 0
index = 1
for word,_ in counter:
  vocabulary[word] = index
  index+=1

def get_vectors(sentence):
  temp = [vocabulary[word] for word in sentence.split() if word in vocabulary]
  vector = [0] * maxlen
  curlen = len(temp)
  if(maxlen-curlen<0):
    vector = temp[:maxlen]
  else:
    vector[maxlen-curlen:] = temp

  return torch.from_numpy(np.asarray(vector,dtype='int32'))

maxlen = 20
embeddim = 300

Xtrain = torch.zeros(trainlen,maxlen)
for i in range(trainlen):
  Xtrain[i] = get_vectors(traincorpus[i])

Xval = torch.zeros(vallen,maxlen)
for i in range(vallen):
  Xval[i] = get_vectors(valcorpus[i])

Xtest = torch.zeros(testlen,maxlen)
for i in range(testlen):
  Xtest[i] = get_vectors(testcorpus[i])

embeddingindex = {}
with open('glove.42B.300d.txt','r',encoding='utf-8') as f:
  for line in f.readlines():
    vectors = line.split()
    word = vectors[0]
    embedding = torch.from_numpy(np.asarray(vectors[1:],'float32'))
    embeddingindex[word] = embedding

embeddingmatrix = torch.zeros(len(vocabulary),embeddim).to(device)
for word,i in list(vocabulary.items()):
  if(word in embeddingindex):
    embeddingmatrix[i] = embeddingindex[word]
  else:
    embeddingmatrix[i] = torch.rand(embeddim)

lamda = 0.1

filtersizes = [3,4,5]

class ABCNN(nn.Module):
    def __init__(self,maxlen,embeddim,embedmatrix,lamda,atthidden=128,channels=50,numclasses=2):
        super(ABCNN,self).__init__()
        self.maxlen = maxlen
        self.embeddim = embeddim
        self.embedmatrix = embedmatrix
        self.atthidden = atthidden
        self.numclasses = numclasses
        self.lamda = lamda
        
        self.embed = nn.Embedding.from_pretrained(self.embedmatrix)
        self.attlinear1 = nn.Linear(self.embeddim*2,self.atthidden)
        self.attlinear2 = nn.Linear(self.atthidden,1)
        
        self.conv1 = nn.Conv1d(self.embeddim*2,channels,filtersizes[0],stride=1)
        self.conv2 = nn.Conv1d(self.embeddim*2,channels,filtersizes[1],stride=1)
        self.conv3 = nn.Conv1d(self.embeddim*2,channels,filtersizes[2],stride=1)
        self.pool1 = nn.MaxPool1d(self.maxlen-filtersizes[0]+1,1,0)
        self.pool2 = nn.MaxPool1d(self.maxlen-filtersizes[1]+1,1,0)
        self.pool3 = nn.MaxPool1d(self.maxlen-filtersizes[2]+1,1,0)
        
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(channels*3,numclasses)
        
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()
    
    def attentionword(self,curembed,outputs,curindex):
        attentionmaps = torch.zeros(outputs.size(0),outputs.size(1)-1).to(device)
        curembeddings = torch.zeros(outputs.size(0),outputs.size(1)-1,self.embeddim).to(device)
        index = 0
        for j in range(self.maxlen):
            if(curindex==j):
                continue
            curembeddings[:,index,:] = outputs[:,j,:]
            embedcon = torch.cat((curembed,outputs[:,index,:]),dim=1)
            embedconout = self.tanh(self.attlinear1(embedcon))
            embedconout = self.attlinear2(embedconout)
            embedconout = math.pow((1-lamda),abs(curindex-index)-1) * embedconout
            attentionmaps[:,index] = embedconout.squeeze(1)
            index+=1

        alpha = self.soft(attentionmaps)
        attentionout = alpha.unsqueeze(2) * curembeddings 
        attentionout = torch.sum(attentionout,dim=1)
        return attentionout
    
    def attentionembed(self,embeddings):
        newembed = torch.zeros(embeddings.size(0),self.maxlen,self.embeddim).to(device)
        for i in range(self.maxlen):
            newembed[:,i,:] = self.attentionword(embeddings[:,i,:],embeddings,i)
        return newembed
      
    def forward(self,x):
        embedout = self.embed(x)
        embedattout = self.attentionembed(embedout)
        embednew = torch.cat((embedout,embedattout),dim=2)
        
        embednew = embednew.transpose(1,2)
        conv1out = self.conv1(embednew)
        conv2out = self.conv2(embednew)
        conv3out = self.conv3(embednew)
        
        pool1out = self.pool1(conv1out)
        pool2out = self.pool2(conv2out)
        pool3out = self.pool3(conv3out)
        
        conout = torch.cat([pool1out,pool2out,pool3out],dim=1)
        conout = conout.view(conout.size(0),-1)
        conout = self.drop(conout)
        conout = self.linear(conout)
        
        return F.log_softmax(conout,1)

attmodel = ABCNN(maxlen,embeddim,embeddingmatrix,lamda).to(device)

x = torch.ones(4,maxlen).long().to(device)
outx = attmodel(x)
print(outx.size())

batchsize = 32

trainarray = torch.utils.data.TensorDataset(Xtrain,ytrain)
trainloader = torch.utils.data.DataLoader(trainarray,batchsize)

valarray = torch.utils.data.TensorDataset(Xval,yval)
valloader = torch.utils.data.DataLoader(valarray,batchsize)

testarray = torch.utils.data.TensorDataset(Xtest,ytest)
testloader = torch.utils.data.DataLoader(testarray,batchsize)

numepochs = 25

def get_accuracy(model,loader):
  acc = 0
  total = 0
  model.eval()
  with torch.no_grad():
    for indices,labels in loader:
      indices = indices.long().to(device)
      labels = labels.long().to(device)
      
      total+=indices.size(0)
      output = model(indices)
      output = torch.max(output,1)[1]
      acc+=torch.sum(labels==output).item()
      
    return ((acc/total)*100)

def get_loss(model,loader):
  curloss = 0.0
  model.eval()
  with torch.no_grad():
    for indices,labels in loader:
      indices = indices.long().to(device)
      labels = labels.long().to(device)
      
      output = model(indices)
      curloss+=F.nll_loss(output,labels)
      
    return (curloss/len(loader))

optimizer = optim.Adam(attmodel.parameters(),lr=0.001)
#loss = nn.CrossEntropyLoss()

bestloss = np.Inf
best_model_wts = copy.deepcopy(attmodel.state_dict())
for epoch in range(numepochs):
  attmodel.train()
  epochloss = 0.0
  epochacc = 0
  for indices,labels in trainloader:
      indices = indices.long().to(device)
      labels = labels.long().to(device)
      
      outputs = attmodel(indices)
      criterion = F.nll_loss(outputs,labels)
      
      epochloss+=criterion.item()
      criterion.backward()
      optimizer.step()
      
  curloss = get_loss(attmodel,valloader)
  curacc = get_accuracy(attmodel,valloader)
  curtrainacc = get_accuracy(attmodel,trainloader)
  print("Epoch {} Train Accuracy {} ValLoss {} Val Accuracy {} ".format(epoch+1,curtrainacc,curloss,curacc))
  if(curloss<bestloss):
    bestloss = curloss
    best_model_wts = copy.deepcopy(attmodel.state_dict())

attmodel.load_state_dict(best_model_wts)
testacc = get_accuracy(attmodel,testloader)
print("Test Accuracy {} ".format(testacc))

