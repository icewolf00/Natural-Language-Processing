import pandas as pd
import numpy as np
from sklearn.linear_model  import LogisticRegression

from nltk.tokenize import word_tokenize
import spacy
global word_to_ix


if __name__ == '__main__':
	train=pd.read_csv('olid-training-v1.0.tsv', sep='\t')
	data = train.values
	print(data.shape)
	nlp = spacy.load("en_core_web_lg")
	vocab = {}
	label = np.zeros((13240,))
	cnt = 0
	#mx = 0
	#mxlen 158
	word_to_ix = {}
	data2 = np.zeros((13240,300))
	for x in range(data.shape[0]):
		if x%1000 == 0:
			print(x)
		# if x==10000:
		# 	break
		tmp = data[x][1]
		#print(tmp)
		doc = nlp(tmp)
		if data[x][4] == 'IND':
			label[x] = 1
		if data[x][4] == 'GRP':
			label[x] = 2
		#print(doc.vector.shape)
		data2[x] = doc.vector
	print('start LogisticRegression')
	lr=LogisticRegression()
	lr.fit(data2,label)
	print(lr.coef_)
	test=pd.read_csv('testset-levelc.tsv', sep='\t')
	data = test.values
	f = open('logc.csv','w')
	for x in range(data.shape[0]):
		if x%1000 == 0:
			print(x)
		tmp = data[x][1]
		#print(tmp)
		ind = data[x][0]
		doc = nlp(tmp)
		inp = doc.vector
		inp = np.reshape(inp,(1,300))
		res = np.round(lr.predict(inp))
		print(res)
		if res ==1:
			print(str(ind)+','+'IND',file=f)
		elif res ==2:
			print(str(ind)+','+'GRP',file=f)
		else:
			print(str(ind)+','+'OTH',file=f)
