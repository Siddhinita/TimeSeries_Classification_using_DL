import keras
import os
import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
class final_accuracy(keras.callbacks.Callback):
	def __init__(self,x_test2,y_pred1,y_true1,y_true2):
		super(keras.callbacks.Callback, self).__init__()
		self.x_test2 = x_test2
		self.y_pred1 = y_pred1
		self.y_true1 = y_true1
		self.y_true2 = y_true2
	def on_epoch_end(self, epoch, logs={}):
		y_pred2 = self.model.predict(self.x_test2)
		y_pred2 = np.argmax(y_pred2,axis=1)
		acc = 0
		for i in range(y_pred2.shape[0]):
			if y_pred2[i] == self.y_true2[i]:
				if y_pred2[i] ==0 :
					acc+=1
				elif self.y_pred1[i] == self.y_true1[i]:
					acc+=1
		print(acc/y_pred2.shape[0])
		f = open('acc_record.txt','a+')
		f.write(str(acc/y_pred2.shape[0]))
		f.write('\n')
		f.close()
			
				
				
