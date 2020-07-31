import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def create_input_data(il = 24, train_split = 0.7, data_filepath = './../inpocket/'):
	#if os.path.isfile('./inpocket_data3.npy'):
	#	return np.load('./inpocket_data.npy')
	files = os.listdir(data_filepath)
	X_train = []
	Y_train_i = []
	Y_train = []
	X_valid = []
	Y_valid_i = []
	Y_valid = []
	columns = [ 'X AXIS','Y AXIS', 'Z AXIS', 'ACC MAGNITUDE', 'X FILTER AXIS', 'Y FILTER AXIS',
        'Z FILTER AXIS', 'XG_AXIS', 'YG_AXIS', 'ZG_AXIS', 'Gyro_filter_X',
       'Gyro_filter_Y', 'Gyro_filter_Z','x_magno',
       'y_magno', 'z_magno', 'MAGNETOMETER']	
	for i,file_ in enumerate(files):
		data = pd.read_csv(data_filepath + file_)
		data['MODE'] = data['MODE'].replace('High',6)
		data['MODE'] = data['MODE'].replace('Low',6)
		cur = -1
		st = 0
		end = 0 
		for c,row in data.iterrows():
			if cur == -1 and pd.isna(row['MODE']) is False:
				cur = row['MODE']
				st = c
			if row['MODE'] != cur and pd.isna(row['MODE']) is False:
				end = c
				#split data
				train_length = int((end-st) * train_split)
				valid_length = end-st-train_length
				if train_length >= il:
					for i in range(st,st + train_length - il):
						X_train.append(np.array(data.ix[i:i+il-1,columns]))
						Y_train_i.append(cur)
				#if train_length >= il and valid_length < il
				if valid_length >= il:
					for i in range(st + train_length,end - il):
						X_valid.append(np.array(data.ix[i:i+il-1,columns]))
						Y_valid_i.append(cur)

				st = end
				cur = row['MODE']
	X_train = np.array(X_train)
	#X_train = X_train[:,np.newaxis,:,:]
	Y_train = np.zeros((len(Y_train_i),5))
	Y_train_i = np.array(Y_train_i,dtype='int')
	Y_train[np.arange(len(Y_train_i)),Y_train_i - 1] = 1
	X_valid = np.array(X_valid)
	#X_valid = X_valid[:,np.newaxis,:,:]
	Y_valid = np.zeros((len(Y_valid_i),5))
	Y_valid_i = np.array(Y_valid_i,dtype='int')
	Y_valid_i-= 1
	print(Y_valid_i)
	for s in Y_valid_i:
		if s == 5:
			print('ye')
	Y_valid[np.arange(len(Y_valid_i)),Y_valid_i] = 1
	print(X_train.shape)
	print(X_valid.shape[0])
	np.save('./inpocket_data.npy',(X_train, Y_train, X_valid, Y_valid, Y_valid_i))
	return X_train, Y_train, X_valid, Y_valid, Y_valid_i
#""", 'ACC MAGNITUDE', 'X FILTER AXIS', 'Y FILTER AXIS',
        #'Z FILTER AXIS', 'XG_AXIS', 'YG_AXIS', 'ZG_AXIS', 'Gyro_filter_X',
       #'Gyro_filter_Y', 'Gyro_filter_Z', 'BAROMETER', 'HEIGHT', 'x_magno',
       #'y_magno', 'z_magno', 'MAGNETOMETER', 'LATITUDE', 'LONGITUDE']"""
create_input_data()
