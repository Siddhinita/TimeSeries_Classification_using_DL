import numpy as np
import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.abspath(os.path.join('..', 'models')))
from fcn import Classifier_FCN
from create_wq_data import create_wq_data
nb_classes = 2
x_train,y_train,x_test,y_test = create_wq_data(il = 35, train_split = 0.7, data_filepath = './../inpocket/')
print(x_train.shape)
y_true = np.argmax(y_test, axis = 1)
input_shape = x_train.shape[1:]
output_directory = './'

classifier =    Classifier_FCN(output_directory,input_shape, nb_classes, True)

classifier.fit(x_train,y_train,x_test,y_test, y_true)


