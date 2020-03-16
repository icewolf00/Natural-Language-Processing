import pandas as pd
import numpy as np
import re
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import linalg
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

title_list = pickle.load(open("title_list.pickle", "rb"))
title_list_test = pickle.load(open("title_list_test.pickle", "rb"))

#train
# vectorizer = CountVectorizer(max_features = 5000)
# X = vectorizer.fit_transform([x[1] for x in title_list])
# word = vectorizer.get_feature_names()
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(X)
# pickle.dump(vectorizer.vocabulary_, open("vec_train.pickle", "wb"))
# pickle.dump(tfidf, open("tfidf_train.pickle", "wb"))

#test
# loaded_vec = CountVectorizer(vocabulary=pickle.load(open("vec_train.pickle", "rb")))
# word = loaded_vec.get_feature_names()
# transformer_test = TfidfTransformer()
# tfidf_test = transformer.fit_transform(loaded_vec.fit_transform([x[1] for x in title_list_test]))
# pickle.dump(tfidf_test, open("tfidf_test.pickle", "wb"))

tfidf = pickle.load(open("tfidf_train_5000.pickle", "rb"))
tfidf_test = pickle.load(open("tfidf_test_5000.pickle", "rb"))

#LSA
#train
# U,sigma,V = linalg.svd(tfidf.toarray(), full_matrices=False)
# targetDimension = 256
# U2 = U[0:, 0:targetDimension]
# V2 = V[0:targetDimension, 0:]
# sigma2 = np.diag(sigma[0:targetDimension])
# pickle.dump(U2, open("U_train_5000_256.pickle", "wb"))
# pickle.dump(V2, open("V_train_5000_256.pickle", "wb"))
# pickle.dump(sigma2, open("sigma_train_5000_256.pickle", "wb"))

#test
# U,sigma,V = linalg.svd(tfidf_test.toarray(), full_matrices=False)
# targetDimension = 256
# U2 = U[0:, 0:targetDimension]
# V2 = V[0:targetDimension, 0:]
# sigma2 = np.diag(sigma[0:targetDimension])
# pickle.dump(U2, open("U_test_5000_256.pickle", "wb"))
# pickle.dump(V2, open("V_test_5000_256.pickle", "wb"))
# pickle.dump(sigma2, open("sigma_test_5000_256.pickle", "wb"))

U_train_5000_256 = pickle.load(open("U_train_5000_256.pickle", "rb"))
U_test_5000_256 = pickle.load(open("U_test_5000_256.pickle", "rb"))
tid_list = [x[0] for x in title_list]
tid_list_test = [x[0] for x in title_list_test]

train_file = pd.read_csv("train.csv")
train_file.fillna('UNKNOWN', inplace=True)
agreed = train_file[train_file["label"] == "agreed"]
disagreed = train_file[train_file["label"] == "disagreed"]
unrelated = train_file[train_file["label"] == "unrelated"]
agreed_list = agreed[['tid1', 'tid2']].values
disagreed_list = disagreed[['tid1', 'tid2']].values
unrelated_list = unrelated[['tid1', 'tid2']].values

# agreed_concate = []
# for x in tqdm(agreed_list):
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     agreed_concate.append(np.concatenate((embedding1, embedding2)))
# pickle.dump(agreed_concate, open("agreed_concate.pickle", "wb"))
# agreed_concate = pickle.load(open("agreed_concate.pickle", "rb"))

# disagreed_concate = []
# for x in tqdm(disagreed_list):
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     disagreed_concate.append(np.concatenate((embedding1, embedding2)))
# pickle.dump(disagreed_concate, open("disagreed_concate.pickle", "wb"))
# disagreed_concate = pickle.load(open("disagreed_concate.pickle", "rb"))

# unrelated_tids = []
# for x in tqdm(unrelated_list):
#     unrelated_tids.append([tid_list.index(x[0]), tid_list.index(x[1])])

# unrelated_concate = []
# for x in tqdm(unrelated_list):
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     unrelated_concate.append(np.concatenate((embedding1, embedding2)))
# pickle.dump(unrelated_concate, open("unrelated_concate.pickle", "wb"))
# unrelated_concate = pickle.load(open("unrelated_concate.pickle", "rb"))

# similarity    
# agreed_cosine = []
# for x in agreed_list:
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     simi = 1 - spatial.distance.cosine(embedding1, embedding2)
#     agreed_cosine.append(simi)
# agreed_cosine = np.array(agreed_cosine)
# pickle.dump(agreed_cosine, open("agreed_cosine_5000_256.pickle", "wb"))
agreed_cosine = pickle.load(open("agreed_cosine_5000_256.pickle", "rb"))
# disagreed_cosine = []
# for x in disagreed_list:
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     simi = 1 - spatial.distance.cosine(embedding1, embedding2)
#     disagreed_cosine.append(simi)
# disagreed_cosine = np.array(disagreed_cosine)
# pickle.dump(disagreed_cosine, open("disagreed_cosine_5000_256.pickle", "wb"))
disagreed_cosine = pickle.load(open("disagreed_cosine_5000_256.pickle", "rb"))

# unrelated_cosine = []
# for x in tqdm(unrelated_list):
#     embedding1 = U_train_5000_256[tid_list.index(x[0])]
#     embedding2 = U_train_5000_256[tid_list.index(x[1])]
#     simi = 1 - spatial.distance.cosine(embedding1, embedding2)
#     unrelated_cosine.append(simi)
# unrelated_cosine = np.array(unrelated_cosine)
# pickle.dump(unrelated_cosine, open("unrelated_cosine_5000_256.pickle", "wb"))
unrelated_cosine = pickle.load(open("unrelated_cosine_5000_256.pickle", "rb"))

test_file = pd.read_csv("test.csv")
test_file.fillna('UNKNOWN', inplace=True)
test_id_list = test_file[['id']].values
test_title_list = test_file[['tid1', 'tid2']].values

# test_embedding = []
# for x in tqdm(test_title_list):
#     embedding1 = U_test_5000_256[tid_list_test.index(x[0])]
#     embedding2 = U_test_5000_256[tid_list_test.index(x[1])]
#     test_embedding.append([embedding1, embedding2])
# pickle.dump(test_embedding, open("test_embedding.pickle", "wb"))
test_embedding = pickle.load(open("test_embedding.pickle", "rb"))

# test_concate = []
# for x in tqdm(test_embedding):
#     test_concate.append(np.concatenate((x[0], x[1])))
# pickle.dump(test_concate, open("test_concate.pickle", "wb"))
test_concate = pickle.load(open("test_concate.pickle", "rb"))

test_cosine = []
for x in tqdm(test_embedding):
    simi = 1 - spatial.distance.cosine(x[0], x[1])
    test_cosine.append(simi)

result_file = open("result_cosine.csv", "w")
result_file.write("Id,Category\n")
for i in range(len(test_id_list)):
    if test_cosine[i] < 0.27:
        result_file.write(str(test_id_list[i][0])+",unrelated\n")
    elif test_cosine[i] > 0.39:
        result_file.write(str(test_id_list[i][0])+",agreed\n")
    else:
        result_file.write(str(test_id_list[i][0])+",disagreed\n")
result_file.close()
# 0.63489 0.63909

agreed_concate = pickle.load(open("agreed_concate.pickle", "rb"))
disagreed_concate = pickle.load(open("disagreed_concate.pickle", "rb"))
unrelated_concate = pickle.load(open("unrelated_concate.pickle", "rb"))
related_concate = np.concatenate((agreed_concate, disagreed_concate))
related_label = np.ones(len(related_concate))
unrelated_label = np.zeros(len(unrelated_concate))
all_concate = np.concatenate((related_concate, unrelated_concate))
all_label = np.concatenate((related_label, unrelated_label))

relaSVM = LinearSVC(random_state=0, tol=1e-5, verbose=1)
relaSVM.fit(all_concate, all_label)

agreed_label = np.ones(len(agreed_concate))
disagreed_label = np.zeros(len(disagreed_concate))
adisg_concate = np.concatenate((agreed_concate, disagreed_concate))
adisg_label = np.concatenate((agreed_label, disagreed_label))

adisgSVM = LinearSVC(random_state=0, tol=1e-5, verbose=1)
adisgSVM.fit(adisg_concate, adisg_label)

test_svm = []
for x in tqdm(test_concate):
    if relaSVM.predict([x]) == 1:
        if adisgSVM.predict([x]) == 1:
            test_svm.append(1)
        else:
            test_svm.append(-1)
    else:
        test_svm.append(0)

result_file2 = open("result_svm.csv", "w")
result_file2.write("Id,Category\n")
for i in range(len(test_id_list)):
    if test_svm[i] == 0:
        result_file2.write(str(test_id_list[i][0])+",unrelated\n")
    elif test_svm[i] == 1:
        result_file2.write(str(test_id_list[i][0])+",agreed\n")
    else:
        result_file2.write(str(test_id_list[i][0])+",disagreed\n")
result_file2.close()

# validation
#svm
test_agree = np.concatenate((agreed_concate[:5000], disagreed_concate[:5000]))
valid_agree_ans = adisgSVM.predict(test_agree)
true_agree_ans = np.concatenate((np.ones(5000), np.zeros(5000)))
agree_weight = np.concatenate((np.ones(5000)*0.1, np.ones(5000)*0.9))
# svm_agree = accuracy_score(true_agree_ans, valid_agree_ans, sample_weight = agree_weight)
# accuracy_score(true_agree_ans, valid_agree_ans)
test_rela = np.concatenate((test_agree, unrelated_concate[:10000]))
valid_rela_ans = relaSVM.predict(test_rela)
true_rela_ans = np.concatenate((np.ones(10000), np.zeros(10000)))
agree_weight = np.concatenate((np.ones(10000)*0.66, np.ones(10000)*0.33))
# svm_rela = accuracy_score(true_rela_ans, valid_rela_ans, sample_weight = agree_weight)
# accuracy_score(true_rela_ans, valid_rela_ans)
#cosine
valid_agree_ans = []
for x in test_agree:
    e1 = x[0]
    e2 = x[1]
    simi = 1 - spatial.distance.cosine(e1, e2)
    if simi < 0.27:
        valid_agree_ans.append(0)
    elif simi > 0.39:
        valid_agree_ans.append(1)
    else:
        valid_agree_ans.append(-1)
# accuracy_score(true_agree_ans, valid_agree_ans)
valid_rela_ans = []
for x in test_rela:
    e1 = x[0]
    e2 = x[1]
    simi = 1 - spatial.distance.cosine(e1, e2)
    if simi < 0.27:
        valid_rela_ans.append(0)
    elif simi > 0.39:
        valid_rela_ans.append(1)
    else:
        valid_rela_ans.append(-1)
# accuracy_score(true_rela_ans, valid_rela_ans)