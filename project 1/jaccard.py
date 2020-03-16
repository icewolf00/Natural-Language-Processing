from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from sklearn.svm import SVC
import pickle
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    #print(synsets2)
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = 0
        for ss in synsets2:
        	tmp = synset.path_similarity(ss)
        	if tmp == None:
        	 	tmp = 0
        	if tmp > best_score:
        		best_score = tmp
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if count == 0:
    	count=1
    score /= count
    return score
 
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
	data = pd.read_csv('train.csv', quotechar='"', skipinitialspace=True)
	data = data.values

	ground_truth = data[:,7]
	for x in range(len(ground_truth)):
		if ground_truth[x]=='agreed':
			ground_truth[x] = 1
		elif ground_truth[x]=='disagreed':
			ground_truth[x] = 0
		else:
			ground_truth[x] = 2
	data = data[:,5:7]
	flag = np.zeros((ground_truth.shape[0],3))
	for x in range(ground_truth.shape[0]):
		tmp = ground_truth[x]
		flag[x][tmp]=1
	cnt_g = 0
	cnt_di = 0
	cnt_un = 0
	sim_g = 0
	sim_di = 0
	sim_un = 0
	X = np.zeros((10000,2))
	Y = np.zeros((10000,))
	#print(sentence_similarity(data[1][0],data[1][1]))
	cnt = 0
	for x in range(15000):
		tmp = get_jaccard_sim(data[x][0],data[x][1])
		# if (x/1000)==0:
		# 	print(x)
		tmp2 = get_cosine_sim(data[x][0],data[x][1])
		tmp2 = tmp2[0][1]
		X[cnt][0] = tmp
		X[cnt][0] = tmp2
		if(ground_truth[x] == 1):
			Y[cnt] =1
			cnt +=1
		elif(ground_truth[x] == 0):
			Y[cnt] =0
			cnt +=1
	print(cnt)
	X = X[:cnt]
	Y = Y[:cnt]
	clf = SVC(gamma='auto')
	clf.fit(X, Y)
	f = open('clf_agreed.pickle', 'wb') 
	pickle.dump(clf, f)
		#print(tmp)
	# print(sim_g/cnt_g)
	# print(sim_di/cnt_di)
	# print(sim_un/cnt_un)
	#ground_truth = flag