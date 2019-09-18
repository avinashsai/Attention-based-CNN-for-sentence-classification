import os
import re
import sys
import pickle
import time
import random
import copy
from copy import deepcopy
import argparse
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from loader import *
from model import *
from train import *

seed = 1332
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-da','--dataset',type=str,help='dataset(trec or mr or sst5 or sst2)',default='trec')
	parser.add_argument('-ru','--runs',type=int,help='number of runs',default=10)
	parser.add_argument('-lam','--lamda',type=float,help='distance decay',default=0.0)

	args = parser.parse_args()

	if(args.dataset=='trec'):
		senlen = 10
		numclasses = 6
		trainloader,valloader,testloader,embedmatrix = load_trec(senlen)
		testacc = trainmodel(senlen,embedmatrix,numclasses,args.lamda,args.runs,trainloader,valloader,testloader)
		print("Accuracy on TREC Dataset {} ".format(testacc))

	elif(args.dataset=='mr'):
		senlen = 30
		numclasses = 2
		trainloader,valloader,testloader,embedmatrix = load_mr(senlen)
		testacc = trainmodel(senlen,embedmatrix,numclasses,args.lamda,args.runs,trainloader,valloader,testloader)
		print("Accuracy on MR Dataset {} ".format(testacc))

	elif(args.dataset=='sst5'):
		senlen = 20
		numclasses = 5
		trainloader,valloader,testloader,embedmatrix = load_sst5(senlen)
		testacc = trainmodel(senlen,embedmatrix,numclasses,args.lamda,args.runs,trainloader,valloader,testloader)
		print("Accuracy on SST-5 Dataset {} ".format(testacc))

	elif(args.dataset=='sst2'):
		senlen = 20
		numclasses = 2
		trainloader,valloader,testloader,embedmatrix = load_sst2(senlen)
		testacc = trainmodel(senlen,embedmatrix,numclasses,args.lamda,args.runs,trainloader,valloader,testloader)
		print("Accuracy on SST-2 Dataset {} ".format(testacc))
		
if __name__ == "__main__":
    main()