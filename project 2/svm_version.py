import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC, SVC
from wordsegment import load, segment

training_data = pd.read_csv("project2_data/olid-training-v1.0.tsv", "r", delimiter = '\t', encoding="utf-8")
id_list = training_data['id'].values
tweet_list = training_data['tweet'].values
subtask_alist = training_data['subtask_a'].values
subtask_blist = training_data['subtask_b'].values
subtask_clist = training_data['subtask_c'].values
tweet_list_b = tweet_list[~pd.isnull(subtask_blist)]
tweet_list_c = tweet_list[~pd.isnull(subtask_clist)]

word_file = open("all_word.txt", "r", encoding = 'utf-8')
all_word_unique = []
for line in word_file:
	all_word_unique.append(line.strip("\n"))

vector_file = open("w2v.txt", "r", encoding = 'utf-8')
all_vector = []
for line in vector_file:
	temp = line.split(" ")[:-1]
	temp = [float(embed) for embed in temp]
	all_vector.append(temp)
vector_file.close()

word2vec300 = {}
for i in range(len(all_word_unique)):
	word2vec300[all_word_unique[i]] = all_vector[i]

# pickle.dump(word2vec300, open('word2vec300.pickle', 'wb'))
word2vec300 = pickle.load(open('word2vec300.pickle', 'rb'))


# max_count = 0
tweet_embedding = []
for tweet in tweet_list:
	temp = tweet.split(" ")
	# temp = segment(tweet.replace('@USER', ''))
	sentence_embed = np.zeros(300)
	word_count = 0
	for word in temp:
		try:
			sentence_embed += word2vec300[word]
			# sentence_embed += new_word2vec300[word]
			word_count += 1
		except:
			pass
	tweet_embedding.append(sentence_embed / word_count)

# tweet_embedding_b = []
# for tweet in tweet_list_b:
# 	temp = tweet.split(" ")
# 	# temp = segment(tweet.replace('@USER', ''))
# 	sentence_embed = np.zeros(300)
# 	word_count = 0
# 	for word in temp:
# 		try:
# 			sentence_embed += word2vec300[word]
# 			# sentence_embed += new_word2vec300[word]			
# 			word_count += 1
# 		except:
# 			pass
# 	tweet_embedding_b.append(sentence_embed / word_count)

# tweet_embedding_c = []
# for tweet in tweet_list_c:
# 	temp = tweet.split(" ")
# 	# temp = segment(tweet.replace('@USER', ''))
# 	sentence_embed = np.zeros(300)
# 	word_count = 0
# 	for word in temp:
# 		try:
# 			sentence_embed += word2vec300[word]
# 			sentence_embed += new_word2vec300[word]
# 			# word_count += 1
# 		except:
# 			pass
# 	tweet_embedding_c.append(sentence_embed / word_count)

tweet_embedding_b = np.array(tweet_embedding)[~pd.isnull(subtask_blist)]
tweet_embedding_c = np.array(tweet_embedding)[~pd.isnull(subtask_clist)]


subtask_alist_label = []
for label in subtask_alist:
	if label == 'NOT':
		subtask_alist_label.append(-1)
	if label == 'OFF':
		subtask_alist_label.append(1)

subtask_blist_label = []
for label in subtask_blist:
	if pd.isnull(label):
		pass
	if label == 'UNT':
		subtask_blist_label.append(-1)
	if label == 'TIN':
		subtask_blist_label.append(1)

subtask_clist_label = []
for label in subtask_clist:
	if pd.isnull(label):
		pass
	if label == 'GRP':
		subtask_clist_label.append(1)
	if label == 'IND':
		subtask_clist_label.append(2)
	if label == 'OTH':
		subtask_clist_label.append(3)

subtask_aSVM = LinearSVC(random_state=0, tol=1e-5, verbose=1)
subtask_aSVM.fit(tweet_embedding, subtask_alist_label)

subtask_bSVM = LinearSVC(random_state=0, tol=1e-5, verbose=1)
subtask_bSVM.fit(tweet_embedding_b, subtask_blist_label)

subtask_cSVM = LinearSVC(random_state=0, tol=1e-5, verbose=1,multi_class='crammer_singer')
subtask_cSVM.fit(tweet_embedding_c, subtask_clist_label)

testing_data_a = pd.read_csv("project2_data/testset-levela.tsv", "r", delimiter = '\t', encoding="utf-8")
test_a_idlist = testing_data_a['id'].values
test_a_tweetlist = testing_data_a['tweet'].values
test_a_label = []
count = 0
for tweet in test_a_tweetlist:
	temp = tweet.split(" ")
	# temp = segment(tweet.replace('@USER', ''))
	sentence_embed = np.zeros(300)
	word_count = 0
	for word in temp:
		try:
			sentence_embed += word2vec300[word]
			# sentence_embed += new_word2vec300[word]
			word_count += 1
		except:
			pass
	sentence_embed = sentence_embed / word_count
	# if word_count == 0:
	#	 sentence_embed = sentence_embed = np.zeros(300)
	a_label = subtask_aSVM.predict([sentence_embed])
#	 print(a_label[0])
	test_a_label.append(a_label[0])
file_a = open("svm_label_a.csv", 'w')
for i in range(len(test_a_idlist)):
	if test_a_label[i] == -1:
		file_a.write(str(test_a_idlist[i])+','+'NOT'+'\n')
	else:
		file_a.write(str(test_a_idlist[i])+','+'OFF'+'\n')
file_a.close()

testing_data_b = pd.read_csv("project2_data/testset-levelb.tsv", "r", delimiter = '\t', encoding="utf-8")
test_b_idlist = testing_data_b['id'].values
test_b_tweetlist = testing_data_b['tweet'].values
test_b_label = []
count = 0
for tweet in test_b_tweetlist:
#	 print(tweet)
	temp = tweet.split(" ")
	# temp = segment(tweet.replace('@USER', ''))
	sentence_embed = np.zeros(300)
	word_count = 0
	for word in temp:
		try:
			sentence_embed += word2vec300[word]
			# sentence_embed += new_word2vec300[word]
			word_count += 1
		except:
			pass
	sentence_embed = sentence_embed / word_count
	# if word_count == 0:
	#	 sentence_embed = sentence_embed = np.zeros(300)
	b_label = subtask_bSVM.predict([sentence_embed])
#	 print(b_label)
	test_b_label.append(b_label[0])
file_b = open("svm_label_b.csv", 'w')
for i in range(len(test_b_idlist)):
	if test_b_label[i] == -1:
		file_b.write(str(test_b_idlist[i])+','+'UNT'+'\n')
	else:
		file_b.write(str(test_b_idlist[i])+','+'TIN'+'\n')
file_b.close()

testing_data_c = pd.read_csv("project2_data/testset-levelc.tsv", "r", delimiter = '\t', encoding="utf-8")
test_c_idlist = testing_data_c['id'].values
test_c_tweetlist = testing_data_c['tweet'].values
test_c_label = []
count = 0
for tweet in test_c_tweetlist:
	temp = tweet.split(" ")
	# temp = segment(tweet.replace('@USER', ''))
	sentence_embed = np.zeros(300)
	word_count = 0
	for word in temp:
		try:
			sentence_embed += word2vec300[word]
			# sentence_embed += new_word2vec300[word]
			word_count += 1
		except:
			pass
	sentence_embed = sentence_embed / word_count
	# if word_count == 0:
	#	 sentence_embed = sentence_embed = np.zeros(300)
	c_label = subtask_cSVM.predict([sentence_embed])
	test_c_label.append(c_label[0])
file_c = open("svm_label_c.csv", 'w')
for i in range(len(test_c_idlist)):
	if test_c_label[i] == 1:
		file_c.write(str(test_c_idlist[i])+','+'GRP'+'\n')
	else:
		if test_c_label[i] == 2:
			file_c.write(str(test_c_idlist[i])+','+'IND'+'\n')
		else:
			file_c.write(str(test_c_idlist[i])+','+'OTH'+'\n')
file_c.close()


true_a = open('project2_data/labels-levela.csv', 'r')
true_a_label = []
for line in true_a:
	if line.strip('\n').split(',')[1] == 'NOT':
		true_a_label.append(-1)
	else:
		true_a_label.append(1)
svm_a = open('svm_label_a.csv', 'r')
svm_a_label = []
for line in svm_a:
	if line.strip('\n').split(',')[1] == 'NOT':
		svm_a_label.append(-1)
	else:
		svm_a_label.append(1)
sklearn.metrics.f1_score(true_a_label, svm_a_label, average='macro')

true_b = open('project2_data/labels-levelb.csv', 'r')
true_b_label = []
for line in true_b:
	if line.strip('\n').split(',')[1] == 'UNT':
		true_b_label.append(-1)
	else:
		true_b_label.append(1)
svm_b = open('svm_label_b.csv', 'r')
svm_b_label = []
for line in svm_b:
	if line.strip('\n').split(',')[1] == 'UNT':
		svm_b_label.append(-1)
	else:
		svm_b_label.append(1)
sklearn.metrics.f1_score(true_b_label, svm_b_label, average='macro')

true_c = open('project2_data/labels-levelc.csv', 'r')
true_c_label = []
for line in true_c:
	if line.strip('\n').split(',')[1] == 'GRP':
		true_c_label.append(1)
	else:
		if line.strip('\n').split(',')[1] == 'IND':
			true_c_label.append(2)
		else:
			true_c_label.append(3)
svm_c = open('svm_label_c.csv', 'r')
svm_c_label = []
for line in svm_c:
	if line.strip('\n').split(',')[1] == 'GRP':
		svm_c_label.append(1)
	else:
		if line.strip('\n').split(',')[1] == 'IND':
			svm_c_label.append(2)
		else:
			svm_c_label.append(3)
sklearn.metrics.f1_score(true_c_label, svm_c_label, average='macro')