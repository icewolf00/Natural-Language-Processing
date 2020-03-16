from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
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
	#f = open('out.csv','w')
	#print('Id,Category',file=f)
	f = open('clf_agreed.pickle', 'rb')
	clf_agreed = pickle.load(f)
	f = open('clf_related.pickle', 'rb')
	clf_related = pickle.load(f)
	a = np.ones((1,2)) #;a[0][0]=0.1001; a[0][1]=0.02;
	a = clf_related.predict(a)
	f = open('svm.csv','w')
	#print(a)
	print('Id,Category',file=f)
	for x in range(data.shape[0]):
		tmp1 = get_jaccard_sim(data[x][0],data[x][1])
		tmp2 = get_cosine_sim(data[x][0],data[x][1])
		tmp2 = tmp2[0][1]
		tmp = np.zeros((1,2))
		tmp[0][0] = tmp1; tmp[0][1] = tmp2;
		result1 = clf_related.predict(tmp)
		result2 = clf_agreed.predict(tmp)
		if result1 == 0:
			print(str(did[x])+","+'unrelated',file = f)
		elif result2 == 0:
			print(str(did[x])+","+'disagreed',file = f)
		else:
			print(str(did[x])+","+'agreed',file = f)