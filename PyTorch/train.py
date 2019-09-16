import copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numepochs = 25

def evaluate(model,loader):
    curloss = 0.0
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for indices,labels in loader:
            indices = indices.long().to(device)
            labels = labels.to(device)

            total+=indices.size(0)
            output = model(indices)
            loss = F.cross_entropy(output,labels,reduction='sum')
            curloss+=loss.item()
            output = torch.max(output,1)[1]
            acc+=torch.sum(labels==output).item()

        return (curloss/total),((acc/total)*100)

def trainmodel(senlen,embedmatrix,numclasses,lamda,runs,trainloader,valloader,testloader):
    testaccuracy = []
    avgtest = 0.0
    for run in range(runs):
        attmodel = ABCNN(senlen,embedmatrix,numclasses,lamda).to(device)
        optimizer = optim.Adam(attmodel.parameters(),lr=0.001)
        bestloss = np.Inf
        best_model_wts = copy.deepcopy(attmodel.state_dict())
        attmodel.train()
        for epoch in range(numepochs):
            attmodel.train()
            for indices,labels in trainloader:
                indices = indices.long().to(device)
                labels = labels.to(device)

                attmodel.zero_grad()
                outputs = attmodel(indices)
                loss = F.cross_entropy(outputs,labels)

                loss.backward()
                optimizer.step()

            curvalloss,curvalacc = evaluate(attmodel,valloader)
            
            if(curvalloss<bestloss):
                bestloss = curvalloss
                best_model_wts = copy.deepcopy(attmodel.state_dict())

        attmodel.load_state_dict(best_model_wts)
        _,testacc = evaluate(attmodel,testloader)
        avgtest+=testacc
    
    avgtest = avgtest/runs
    return avgtest