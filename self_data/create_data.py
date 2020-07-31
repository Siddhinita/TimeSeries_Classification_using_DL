import pandas as pd
import numpy as np
import os
import warnings
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
print("hi")
#up - 0
#down - 1
#walk - 2

#if os.path.isfile('./data.npy'):
#	return np.load('./data.npy')

train_split = 0.7
il = 100
X_train = []
Y_train = []
X_valid = []
Y_valid = []
columns = [ 'X','Y', 'Z']
for s,val in zip(['./data/up/','./data/down/','./data/walk/'],[[1,0,0],[0,1,0],[0,0,1]]):
	l = os.listdir(s)
	for a in l:
		data = pd.read_csv(s + a,engine='python')
		print(len(data))
		train_length = int(len(data) * train_split)
		valid_length = len(data) - train_length
		if train_length >= il:
			for i in range(0,train_length - il):
				X_train.append(np.array(data.ix[i:i+il-1,columns]))
				Y_train.append(val)
		if valid_length >= il:
			for i in range(train_length,len(data)- il):
				X_valid.append(np.array(data.ix[i:i+il-1,columns]))
				Y_valid.append(val)
X_train,Y_train = shuffle(X_train, Y_train, random_state=0)
X_valid,Y_valid = shuffle(X_valid, Y_valid, random_state=0)
print(len(X_train))
print(len(X_valid))
np.save('./data.npy',[X_train,Y_train,X_valid,Y_valid])
	#return X_train,Y_train,X_valid,Y_valid

