import wordsegment
import emoji
import pandas as pd
from wordsegment import load, segment
import pickle
if __name__ == '__main__':
	train=pd.read_csv('olid-training-v1.0.tsv', sep='\t')
	data = train.values
	print(data.shape)
	load()
	al = []
	fil = open('data.pickle', 'wb')

	for x in range(data.shape[0]):
		if x %1000 ==0:
			print(x)
		tmp_ls = []
		tmp = data[x][1]
		tmp = emoji.demojize(tmp)
		ls = segment(tmp)
		tmp_ls.append(data[x][0])
		cnt = 0
		for y in ls:
			if y== 'user':
				cnt +=1
		while cnt >3:
			ls.remove('user')
			cnt -= 1
		#print(ls)
		tmp_ls.append(ls)
		tmp_ls.append(data[x][2])
		tmp_ls.append(data[x][3])
		tmp_ls.append(data[x][4])
		#print(tmp_ls)
		al.append(tmp_ls)
	pickle.dump(al, fil)
	fil.close()