from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)
    
def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()
if __name__ == '__main__':
	data = pd.read_csv('test.csv', quotechar='"', skipinitialspace=True)
	data = data.values
	did = data[:,0]
	data = data[:,5:7]
	f = open('out.csv','w')
	print('Id,Category',file=f)
	for x in range(data.shape[0]):
		#tmp = get_jaccard_sim(data[x][0],data[x][1])
		tmp = get_cosine_sim(data[x][0],data[x][1])
		tmp = tmp[0][1]
		if tmp > 0.4 and ('rumors' in data[x][0] or 'rumors' in data[x][1]):
			print(str(did[x])+","+'disagreed',file = f)
		elif tmp < 0.4:
			print(str(did[x])+","+'unrelated',file = f)
		else:
			print(str(did[x])+","+'agreed',file = f)