#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH -o DL_hw8.out
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=16000

import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#Loading saved model
yaml_file = open('merge_model.yaml', 'r')  # Load the model you wish 
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)

# load weights into new model
model.load_weights("weights_merge.hdf5")
print("Loaded model from disk")
model.summary()

test_img_path = 'data/Test_set/'
img_list = os.listdir(test_img_path)

for img_no in range(len(img_list)):
    img_name = test_img_path+str(img_no)+'.png'
    
    img = cv2.imread(img_name)
    org = cv2.resize(img, (640,480))
    proc_img = cv2.resize(img, (256,192))
    proc_img = np.array(proc_img)
    proc_img = np.expand_dims(proc_img, axis=0)
    proc_img = proc_img.astype('float32')
    proc_img = proc_img/255.0
    
    y_pred = model.predict(proc_img)[0][0] 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(org,'Predicted Value:',(10,380), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(org,str(y_pred),(10,430), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Detection', org)
    cv2.waitKey(30) & 0xFF

cv2.destroyAllWindows()


