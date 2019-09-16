import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

embeddim = 300
atthidden = 128
channels = 100
filtersizes = [3,4,5]

class ABCNN(nn.Module):
    def __init__(self,maxlen,embeddingmatrix,numclasses,lamda):
        super(ABCNN,self).__init__()
        self.maxlen = maxlen
        self.embeddim = embeddim
        self.embedmatrix = embeddingmatrix
        self.atthidden = atthidden
        self.numclasses = numclasses
        self.lamda = lamda
        
        self.embed = nn.Embedding.from_pretrained(self.embedmatrix,freeze=True)
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
  
    
    def attentionword(self,curembed,outputs,curindex):
        inp1 = outputs[:,curindex,:].unsqueeze(1).repeat(1,self.maxlen-1,1)
        inp2 = torch.cat([outputs[:,:curindex,:],outputs[:,curindex+1:,:]],dim=1)
        
        inp = torch.cat([inp1,inp2],dim=2)
        
     
        inp = torch.reshape(inp,(-1,self.embeddim*2))
        attout = self.tanh(self.attlinear1(inp)).view(-1,self.maxlen-1,self.atthidden)
        
        if(curindex==0):
			score = torch.stack([math.pow((1-self.lamda),abs(curindex-index)-1) * attout[:,index-1] for index in range(curindex+1,self.maxlen)],1)
        elif(curindex==self.maxlen-1):
			score = torch.stack([math.pow((1-self.lamda),abs(curindex-index)-1) * attout[:,index] for index in range(0,curindex)],1)
        else:
			score1 = torch.stack([math.pow((1-self.lamda),abs(curindex-index-1)) * attout[:,index] for index in range(0,curindex)],1)
			score2 = torch.stack([math.pow((1-self.lamda),abs(curindex-index-1)) * attout[:,index-1] for index in range(curindex+1,self.maxlen)],1)
			score = torch.cat([score1,score2],1)
          
        attout = self.attlinear2(score).squeeze(2)
        alpha = F.softmax(attout,1)
        
        out = alpha.unsqueeze(2) * inp2
        out = torch.sum(out,1)
        return out
    
    def attentionembed(self,embeddings):
        newembed = torch.stack([self.attentionword(embeddings[:,i,:],embeddings,i) for i in range(self.maxlen)],dim=1)
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
        
        return conout