import string
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from convert import *


datapath = '../Datasets/'

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return " ".join(word.lower() for word in words)

def load_trec(senlen):
    lab = {'DESC':0,'ENTY':1,'ABBR':2,'HUM':3,'LOC':4,'NUM':5}
    traincorpus = []
    trainlabels = []
    with open(datapath+'TREC/'+'train.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split(':')
            trainlabels.append(int(lab[words[0]]))
            traincorpus.append(preprocess(" ".join(word for word in words[1].split())))


    testcorpus = []
    testlabels = []
    with open(datapath+'TREC/'+'test.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split(':')
            testlabels.append(int(lab[words[0]]))
            testcorpus.append(preprocess(" ".join(word for word in words[1].split())))

    trainlabels = np.asarray(trainlabels)
    testlabels = np.asarray(testlabels)

    traincorpus,valcorpus,trainlabels,vallabels = train_test_split(traincorpus,trainlabels,test_size=0.1,random_state=42)

    vocab = get_vocab(traincorpus)
    trainloader = get_loader(traincorpus,trainlabels,vocab,senlen)
    valloader = get_loader(valcorpus,vallabels,vocab,senlen)
    testloader = get_loader(testcorpus,testlabels,vocab,senlen)

    embedmatrix = get_embeddings(vocab)

    return trainloader,valloader,testloader,embedmatrix


def load_mr(senlen):
    corpus = []
    poscount = 0
    negcount = 0
    with open(datapath+'MR/rt-polarity.pos','r',encoding='latin1') as f:
        for line in f.readlines():
            corpus.append(preprocess(line[:-1]))
            poscount+=1

    with open(datapath+'MR/rt-polarity.neg','r',encoding='latin1') as f:
        for line in f.readlines():
            corpus.append(preprocess(line[:-1]))
            negcount+=1

    labels = np.zeros(poscount+negcount)
    labels[:poscount] = 1

    corpus,labels = shuffle(corpus,labels,random_state=0)

    traincorpus,testcorpus,trainlabels,testlabels = train_test_split(corpus,labels,test_size=0.2,random_state=42)
    traincorpus,valcorpus,trainlabels,vallabels = train_test_split(traincorpus,trainlabels,test_size=0.1,random_state=42)

    vocab = get_vocab(traincorpus)
    trainloader = get_loader(traincorpus,trainlabels,vocab,senlen)
    valloader = get_loader(valcorpus,vallabels,vocab,senlen)
    testloader = get_loader(testcorpus,testlabels,vocab,senlen)

    embedmatrix = get_embeddings(vocab)

    return trainloader,valloader,testloader,embedmatrix


def load_sst2(senlen):
    traincorpus = []
    trainlabels = []
    with open(datapath+'SST-2/'+'train.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            trainlabels.append(int(words[0]))
            traincorpus.append(preprocess(" ".join(word for word in words[1:])))
    
    valcorpus = []
    vallabels = []
    with open(datapath+'SST-2/'+'dev.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            vallabels.append(int(words[0]))
            valcorpus.append(preprocess(" ".join(word for word in words[1:])))


    testcorpus = []
    testlabels = []
    with open(datapath+'SST-2/'+'test.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            testlabels.append(int(words[0]))
            testcorpus.append(preprocess(" ".join(word for word in words[1:])))

    trainlabels = np.asarray(trainlabels)
    vallabels = np.asarray(vallabels)
    testlabels = np.asarray(testlabels)

    vocab = get_vocab(traincorpus)
    trainloader = get_loader(traincorpus,trainlabels,vocab,senlen)
    valloader = get_loader(valcorpus,vallabels,vocab,senlen)
    testloader = get_loader(testcorpus,testlabels,vocab,senlen)

    embedmatrix = get_embeddings(vocab)

    return trainloader,valloader,testloader,embedmatrix

def load_sst5(senlen):
    traincorpus = []
    trainlabels = []
    with open(datapath+'SST-5/'+'train.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            trainlabels.append(int(words[0]))
            traincorpus.append(preprocess(" ".join(word for word in words[1:])))
    
    valcorpus = []
    vallabels = []
    with open(datapath+'SST-5/'+'dev.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            vallabels.append(int(words[0]))
            valcorpus.append(preprocess(" ".join(word for word in words[1:])))


    testcorpus = []
    testlabels = []
    with open(datapath+'SST-5/'+'test.txt','r',encoding='latin1') as f:
        for line in f.readlines():
            words = line.split()
            testlabels.append(int(words[0]))
            testcorpus.append(preprocess(" ".join(word for word in words[1:])))

    trainlabels = np.asarray(trainlabels)
    vallabels = np.asarray(vallabels)
    testlabels = np.asarray(testlabels)

    vocab = get_vocab(traincorpus)
    trainloader = get_loader(traincorpus,trainlabels,vocab,senlen)
    valloader = get_loader(valcorpus,vallabels,vocab,senlen)
    testloader = get_loader(testcorpus,testlabels,vocab,senlen)

    embedmatrix = get_embeddings(vocab)

    return trainloader,valloader,testloader,embedmatrix