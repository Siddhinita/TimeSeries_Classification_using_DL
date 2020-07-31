import numpy as np
import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.abspath(os.path.join('..', 'models')))
from fcn import Classifier_FCN
from create_qsw_data import create_qsw_data

nb_classes = 3
output_directory = './output/'
x_train,y_train,x_test,y_test = create_qsw_data(il = 34, train_split = 0.8, data_filepath = '/media/sid/Data/HAR/inpocket/')
print("Training on {} samples".format(len(x_train)))
print("Validation on {} samples".format(len(x_test)))
	
input_shape = x_train.shape[1:]
output_directory = './output/'
if os.path.isdir('./output') is False:
	os.mkdir('./output')
y_true = np.argmax(y_test,axis=1)
#print(y_true)
#classifier =    Classifier_RESNET(output_directory,input_shape, nb_classes, True)
classifier =    Classifier_FCN(output_directory,input_shape, nb_classes, True)

classifier.fit(x_train,y_train,x_test,y_test, y_true)


