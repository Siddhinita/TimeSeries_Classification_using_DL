import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def create_wq_data(il = 35, train_split = 0.7, data_filepath = './../inpocket/'):
	if os.path.isfile('./wq_clf.npy'):
		   return np.load('./wq_clf.npy')
	files = os.listdir(data_filepath)
	X_train = []
	Y_train = []
	Y_train = []
	X_valid = []
	Y_valid = []
	Y_valid = []
	columns = [ 'X AXIS','Y AXIS', 'Z AXIS', 'ACC MAGNITUDE', 'X FILTER AXIS', 'Y FILTER AXIS',
        'Z FILTER AXIS', 'XG_AXIS', 'YG_AXIS', 'ZG_AXIS', 'Gyro_filter_X',
       'Gyro_filter_Y', 'Gyro_filter_Z','x_magno',
       'y_magno', 'z_magno', 'MAGNETOMETER']
	for i,file_ in enumerate(files):
		print(i)
		data = pd.read_csv(data_filepath + file_)
		data['MODE'] = data['MODE'].replace('High',6)
		data['MODE'] = data['MODE'].replace('Low',6)
		cur = -1
		st = 0
		end = 0 
		for c,row in data.iterrows():
			if cur == -1 and pd.isna(row['MODE']) is False and row['MODE'] in [3,5]:
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
							X_train.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train.append([0,1])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid.append([0,1])
				else:
					if train_length >= il:
						for i in range(st,st + train_length - il):
							X_train.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train.append([1,0])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid.append([1,0])

					
				st = end
				if row['MODE'] in [3,5]:
					cur = row['MODE']
				else:
					cur = -1
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	Y_train = np.array(Y_train)
	X_valid = np.array(X_valid)
	Y_valid = np.array(Y_valid)
	Y_valid = np.array(Y_valid)

	
	np.save('./wq_clf.npy',(X_train, Y_train, X_valid, Y_valid))
	
	return np.load('./wq_clf.npy')
#create_swq_data(il = 35, train_split = 0.7, data_filepath = '/media/sid/Data/forecast/inpocket_code/inpocket')
    
