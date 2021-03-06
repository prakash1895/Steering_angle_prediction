#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH -o DL_merge.out
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=32000

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import *
from keras.layers import Flatten
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

fh = open('python_output_merge.txt', 'a')

train_data_path = '/home/student001/DL_project/data_new/Train_set/'
test_data_path = '/home/student001/DL_project/data/Test_set/'
csv_file_path = '/home/student001/DL_project/data_new/csv_files/'

train_data_dir_list = os.listdir(train_data_path)

frame_rate = 1

def batch_data_generator (x, y, batch_size):
    while True:
        x_data = np.empty((0,192,256,3))
        y_data = np.array([])
        start = np.random.randint(0, len(y)-batch_size)
        end = min(start + batch_size, len(y))
        for i in np.arange(start, end):
            img = Image.open(x[i][0])
            img_arr = np.array(img)
            img_arr = img_arr.astype('float32')
            img_arr = img_arr/255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            x_data = np.append(x_data, img_arr, axis=0)
            y_data = np.append(y_data, y[i][0])
            img.close()
        yield x_data, y_data
    
print ('Adding the images ...', file=fh)

img_data_list = []
label = []

for idx,data_dir in enumerate(train_data_dir_list):
    print ('Reading directory '+ data_dir, file=fh)
    img_list = os.listdir(train_data_path+data_dir)
    steer_arr = pd.read_csv(csv_file_path+data_dir+'.csv', header=None)
    steer_arr = np.array(steer_arr)
    for img in range(len(img_list)):
        if (img%frame_rate) == 0:
            steer_ang =  float(steer_arr[img])
            img_name = train_data_path+data_dir+'/'+str(img)+'.png'
            img_data_list.append(img_name)
            label.append(steer_ang)
            
img_data_list = np.array(img_data_list).reshape(len(img_data_list),1)
label = np.array(label).reshape(len(label),1)
label = label.astype('float64')

y_min = np.amin(label)
y_max = np.amax(label)

print(y_min, file=fh)
print(y_max, file=fh)

y = label
x = img_data_list
x,y = shuffle(x, y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=2)

np_epochs = 5
batch_size = 32
input_shape = (192,256,3)

def commai_model(input_layer):
    
    model = Conv2D(16, 8, 8, input_shape = input_shape, subsample=(4, 4), border_mode="same", activation = 'relu')(input_layer)
    model = Dropout(0.5)(model)
    model = Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    return Model(input = input_layer, output = model)

def nvidia_model(input_layer):
    
    model = Conv2D (24, 5, 5, input_shape=input_shape, subsample=(2,2), border_mode='same', activation='relu')(input_layer)
    model = Dropout(0.5)(model)
    model = Conv2D (36, 5, 5, subsample=(2,2), border_mode='same', activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Conv2D (48, 5, 5, subsample=(2,2), border_mode='same', activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Conv2D (64, 3, 3, subsample=(2,2), border_mode='same', activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Conv2D (64, 3, 3, subsample=(2,2), border_mode='same', activation = 'relu')(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)
    return Model(input= input_layer, output = model)

input_layer = Input(shape = input_shape)
model_1 = commai_model(input_layer)
model_2 = nvidia_model(input_layer)

x_merged = merge([model_1.output, model_2.output], mode='concat', concat_axis=1)
x_merged = Dense (100, activation = 'relu')(x_merged)
x_merged = Dense (50, activation = 'relu')(x_merged)
x_merged = Dense (10, activation = 'relu')(x_merged)
x_merged = Dropout(0.5)(x_merged)
x_merged = Dense(1)(x_merged)

model = Model(input = input_layer, output = x_merged)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

model_yaml = model.to_yaml()
with open('merge_model.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
print ('model.yaml saved to disk')

filepath="weights_merge.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

train_samples = X_train.shape[0]
val_samples = X_val.shape[0]
test_samples = X_test.shape[0]

samples_per_epoch = train_samples - train_samples % batch_size
nb_val_samples = val_samples - val_samples % batch_size

print ('Training begins...', file=fh)
hist = model.fit_generator(
        batch_data_generator(X_train, y_train, batch_size),
        samples_per_epoch = samples_per_epoch,
        nb_epoch = np_epochs,
        verbose = 1,
        validation_data = batch_data_generator(X_val, y_val, batch_size),
        nb_val_samples = nb_val_samples,
        callbacks = callbacks_list)

print ('Training Ends...', file=fh)

# visualizing losses and accuracy

xc = range(np_epochs)
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

history_data = np.zeros((np_epochs,5))
history_data[:,0] = xc
history_data[:,1] = train_loss
history_data[:,2] = val_loss
history_data[:,3] = train_acc
history_data[:,4] = val_acc

np.savetxt('history_merge.csv', history_data, delimiter=',')

score = model.evaluate_generator(batch_data_generator(X_test, y_test, batch_size), test_samples)
print('Test Loss:', score[0], file=fh)
print('Test accuracy:', score[1], file=fh)

fh.close()



