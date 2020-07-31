import keras 
import numpy as np 
import pandas as pd 
import time 
import os
from utils import save_logs

class Classifier2:

	def __init__(self, output_directory, input_shape, nb_classes,acc, verbose=False):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes,acc)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init_q.hdf5')
	def build_model(self, input_shape, nb_classes,acc):
		input_layer = keras.layers.Input(input_shape)
		output_layer = keras.layers.Dense(nb_classes,activation = 'softmax')(input_layer)
		
		model = keras.models.Model(inputs = input_layer, outputs = output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model_q.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint,acc]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true): 
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 16
		nb_epochs = 40

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks,shuffle = True)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model_q.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)
		if not os.path.exists('./clf2/'):
			os.makedirs('./clf2/')
		save_logs('./clf2/', hist, y_pred, y_true, duration)

		keras.backend.clear_session()

