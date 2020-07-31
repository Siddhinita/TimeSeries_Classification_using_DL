import os
import pandas as pd
import numpy as np
import warnings
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
#queue = [1,0,0]
#stand = [0,1,0]
#walk = [0,0,1]
def create_qsw_data(il = 35, train_split = 0.7, data_filepath = './../inpocket/'):

	if os.path.isfile('./qsw_data.npy'):
		   return np.load('./qsw_data.npy')
	files = os.listdir(data_filepath)
	X_train_s = []
	Y_train_s = []
	X_valid_s = []
	Y_valid_s = []
	X_train_q = []
	Y_train_q = []
	X_valid_q = []
	Y_valid_q = []
	X_train_w = []
	Y_train_w = []
	X_valid_w = []
	Y_valid_w = []

	columns = [ 'X AXIS','Y AXIS', 'Z AXIS']
	for i,file_ in enumerate(files):
		print(i)
		data = pd.read_csv(data_filepath + file_)
		data['MODE'] = data['MODE'].replace('High',6)
		data['MODE'] = data['MODE'].replace('Low',6)
		cur = -1
		st = 0
		end = 0 
		for c,row in data.iterrows():
			if cur == -1 and pd.isna(row['MODE']) is False and row['MODE'] in [3,4,5]:
				cur = row['MODE']
				st = c
			if (row['MODE'] != cur and pd.isna(row['MODE']) is False) or (len(data) == c + 1 and row['MODE'] == cur):
				end = c
				#split data
				train_length = int((end-st) * train_split)
				valid_length = end-st-train_length
				if cur == 3:
					if train_length >= il:
						for i in range(st,st + train_length - il):
							X_train_q.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train_q.append([1,0,0])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid_q.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid_q.append([1,0,0])
				elif cur == 4:
					if train_length >= il:
						for i in range(st,st + train_length - il):
							X_train_s.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train_s.append([0,1,0])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid_s.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid_s.append([0,1,0])
				else:				
					if train_length >= il:
						for i in range(st,st + train_length - il):
							X_train_w.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train_w.append([0,0,1])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid_w.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid_w.append([0,0,1])
				st = end
				if row['MODE'] in [3,4,5]:
					cur = row['MODE']
				else:
					cur = -1
	train_len = min(min(len(X_train_q),len(X_train_s)),len(X_train_w))
	#valid_len = min(min(len(X_valid_q),len(X_valid_s)),len(X_valid_w))
	X_train = np.concatenate([X_train_q[:train_len],X_train_s[:train_len],X_train_w[:train_len]],axis=0)
	Y_train = np.concatenate([Y_train_q[:train_len],Y_train_s[:train_len],Y_train_w[:train_len]],axis=0)
	#X_valid = np.concatenate([X_valid_q[:valid_len],X_valid_s[:valid_len],X_valid_w[:valid_len]],axis=0)
	#Y_valid = np.concatenate([Y_valid_q[:valid_len],Y_valid_s[:valid_len],Y_valid_w[:valid_len]],axis=0)
	
	X_valid = np.concatenate([X_valid_q,X_valid_s,X_valid_w],axis=0)
	Y_valid = np.concatenate([Y_valid_q,Y_valid_s,Y_valid_w],axis=0)
	
	X_train,Y_train = shuffle(X_train, Y_train, random_state=0)
	X_valid,Y_valid = shuffle(X_valid, Y_valid, random_state=0)
	np.save('./qsw_data.npy',[X_train,Y_train,X_valid,Y_valid])
	return X_train,Y_train,X_valid,Y_valid
#create_swq_data(il = 35, train_split = 0.7, data_filepath = '/media/sid/Data/forecast/inpocket_code/inpocket')
    
