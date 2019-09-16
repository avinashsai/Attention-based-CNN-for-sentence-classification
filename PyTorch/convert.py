import collections
from collections import Counter
import numpy as np 
import torch
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddim = 300
batchsize = 32

vecpath = ''

def get_vectors(sentence,vocabulary,maxlen):
    temp = [vocabulary[word] for word in sentence.split() if word in vocabulary]
    vector = [0] * maxlen
    curlen = len(temp)
    if(maxlen-curlen<0):
        vector = temp[:maxlen]
    else:
        vector[maxlen-curlen:] = temp

    return torch.from_numpy(np.asarray(vector,dtype='int32'))

def get_vocab(data):
    words = []
    for sentence in data:
        words+=sentence.split()

    counts = Counter(words).most_common()
    counts.insert(0,('<PAD>',0))
    vocabulary = {word:i for i,(word,_) in enumerate(counts)}
    return vocabulary

def get_loader(data,labels,vocabulary,senlen):
    X = torch.stack([get_vectors(sen,vocabulary,senlen) for sen in data])
    y = torch.from_numpy(labels).long()
    array = torch.utils.data.TensorDataset(X,y)
    loader = torch.utils.data.DataLoader(array,batchsize)
    return loader


def get_embeddings(vocabulary):

    embeddingindex = {}
    with open(vecpath+'glove.840B.300d.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            vectors = line.split(' ')
            word = vectors[0]
            embedding = torch.from_numpy(np.asarray(vectors[1:],'float32'))
            embeddingindex[word] = embedding

    embeddingmatrix = torch.zeros(len(vocabulary),embeddim).to(device)
    for i,word in enumerate(list(vocabulary.items())):
        if(word in embeddingindex):
            embeddingmatrix[i] = embeddingindex[word]
        else:
            embeddingmatrix[i] = torch.rand(embeddim)
    
    return embeddingmatrix