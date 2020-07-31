import numpy as np
import sys
import keras
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#from ResNet import Classifier_RESNET
from fcn import Classifier_FCN
from Classifier2 import Classifier2
from keras import backend as K
from create_swq_data import create_swq_data
from final_accuracy import final_accuracy
nb_classes = 2
clf1_data,clf2_data = create_swq_data(il = 35, train_split = 0.7, data_filepath = './../inpocket/')
x_train1,y_train1,x_test1,y_test1 = clf1_data
x_train2,y_train2,x_test2,y_test2 = clf2_data
input_shape1 = x_train1.shape[1:]

output_directory = './results'
os.makedirs(output_directory)
#classifier =    Classifier_RESNET(output_directory,input_shape, nb_classes, True)
clf1 =    Classifier_FCN(output_directory,input_shape1, nb_classes, True)
y_true1 = np.argmax(y_test1, axis = 1)
clf1.fit(x_train1,y_train1,x_test1,y_test1,y_true1)

model = keras.models.load_model('best_model_sw.hdf5')
print(model.summary())
f = K.function([model.input],[model.layers[-2].output])
print(model.layers[-2])
y_pred1 = model.predict(x_test2)
x_train2 = f([x_train2])[0]
x_test2 = f([x_test2])[0]

y_pred1 = np.argmax(y_pred1, axis = 1)
input_shape2 = x_train2.shape[1:]
print(input_shape2)
np.save('clf2_input.npy',(x_train2,y_train2,x_test2,y_test2))
#clf2
y_true2 = np.argmax(y_test2, axis = 1)
acc = final_accuracy(x_test2,y_pred1,y_true1,y_true2)
clf2 = Classifier2(output_directory,input_shape2, nb_classes,acc,True)
clf2.fit(x_train2,y_train2,x_test2,y_test2,y_true2)
