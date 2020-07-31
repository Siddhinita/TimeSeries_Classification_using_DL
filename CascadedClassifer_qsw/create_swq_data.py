import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def create_swq_data(il = 35, train_split = 0.7, data_filepath = './../inpocket/'):
	if os.path.isfile('./s_w_q_clf1.npy'):
		   return np.load('./s_w_q_clf1.npy'),np.load('./s_w_q_clf2.npy')
	files = os.listdir(data_filepath)
	X_train_sw = []
	Y_train_sw1 = []
	Y_train_sw2 = []
	X_valid_sw = []
	Y_valid_sw1 = []
	Y_valid_sw2 = []

	X_train_q = []
	Y_train_q = []
	X_valid_q = []
	Y_valid_q = []
    
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
							Y_train_q.append([1,0])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid_q.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid_q.append([1,0])
				else:
					if train_length >= il:
						for i in range(st,st + train_length - il):
							X_train_sw.append(np.array(data.ix[i:i+il-1,columns]))
							Y_train_sw1.append([1 if cur == 4 else 0, 1 if cur == 5 else 0])
							Y_train_sw2.append([0,1])
		            #if train_length >= il and valid_length < il
					if valid_length >= il:
						for i in range(st + train_length,end - il):
							X_valid_sw.append(np.array(data.ix[i:i+il-1,columns]))
							Y_valid_sw1.append([1 if cur == 4 else 0, 1 if cur == 5 else 0])
							Y_valid_sw2.append([0,1])

					
				st = end
				if row['MODE'] in [3,4,5]:
					cur = row['MODE']
				else:
					cur = -1
	X_train_sw = np.array(X_train_sw)
	Y_train_sw1 = np.array(Y_train_sw1)
	Y_train_sw2 = np.array(Y_train_sw2)
	X_valid_sw = np.array(X_valid_sw)
	Y_valid_sw1 = np.array(Y_valid_sw1)
	Y_valid_sw2 = np.array(Y_valid_sw2)

	X_train_q = np.array(X_train_q)
	Y_train_q = np.array(Y_train_q)
	X_valid_q = np.array(X_valid_q)
	Y_valid_q = np.array(Y_valid_q)

	np.save('./s_w_q_clf1.npy',(X_train_sw, Y_train_sw1, X_valid_sw, Y_valid_sw1))
	np.save('./s_w_q_clf2.npy',(np.concatenate([X_train_sw,X_train_q],axis=0),
	np.concatenate([Y_train_sw2,Y_train_q],axis=0),
	np.concatenate([X_valid_sw,X_valid_q],axis=0),
	np.concatenate([Y_valid_sw2,Y_valid_q],axis=0)))
	return np.load('./s_w_q_clf1.npy'),np.load('./s_w_q_clf2.npy')
#create_swq_data(il = 35, train_split = 0.7, data_filepath = '/media/sid/Data/forecast/inpocket_code/inpocket')
    
