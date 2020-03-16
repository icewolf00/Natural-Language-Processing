import csv
import spacy
import numpy as np
import pandas as pd
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import pickle


if __name__ == '__main__':
	nlp = spacy.load("en")
	ls = []
	for x in range(167564):
		ls.append([])
	data = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
	data = data.values
	#data = data[1::]
	#document = "Read less on TV and read more books. These eight books will help improve EQ and IQ"
	#document = nlp(document)
	f1 = open('parse.pickle', 'wb')
	# mx = -1
	# for x in range(data.shape[0]):
	# 	tid = max(data[x][1],data[x][2])
	# 	if tid > mx:
	# 		mx = tid
	# print(mx)
	#ls.append(document)
	flag = set()
	for x in range(data.shape[0]):
		if x %10000 == 0:
			print(x)
		tid = data[x][1]
		#print(data[x])
		if tid not in flag:
			flag.add(tid)
			title = data[x][5]
			document = nlp(title)
			#print(title)
			ls[tid] = (tid,document)
		tid = data[x][2]
		if tid not in flag:
			flag.add(tid)
			title = data[x][6]
			document = nlp(title)
			#print(title)
			ls[tid] = (tid,document)
	pickle.dump(ls, f1)
	f1.close()