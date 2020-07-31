import numpy as np
import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.abspath(os.path.join('..', 'models')))
from ResNet import Classifier_RESNET
from fcn import Classifier_FCN
from create_input_data import create_input_data
nb_classes = 5
x_train,y_train,x_test,y_test, y_true = create_input_data(il = 24, train_split = 0.7, data_filepath = './../inpocket/')
print(x_train.shape)
input_shape = x_train.shape[1:]
output_directory = './results'
#classifier =    Classifier_RESNET(output_directory,input_shape, nb_classes, True)
classifier =    Classifier_FCN(output_directory,input_shape, nb_classes, True)

classifier.fit(x_train,y_train,x_test,y_test, y_true)


