

import sys
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

new_img_path = 'data_new_1/Train_set/'
new_csv_path = 'data_new_1/csv_files/'

if not os.path.exists(new_csv_path):
    os.makedirs(new_csv_path)
    
train_data_path = 'data/Train_set/'
csv_file_path = 'data/csv_files/'

train_data_dir_list = os.listdir(train_data_path)

for data_dir in train_data_dir_list:
    if not os.path.exists(new_img_path+data_dir):
        os.makedirs(new_img_path+data_dir)

label = []
for idx,data_dir in enumerate(train_data_dir_list):
    print ('Reading directory '+ data_dir)
    img_list = os.listdir(train_data_path+data_dir)
    steer_arr = pd.read_csv(csv_file_path+data_dir+'.csv', header=None)
    steer_arr = np.array(steer_arr)
    label[:] = []
    k = 0
    for i in range(len(img_list)):
        steer_ang =  float(steer_arr[i])
        img_name = train_data_path+data_dir+'/'+str(i)+'.png'
        img = Image.open(img_name)
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(new_img_path+data_dir+'/'+str(k)+'.png')
        k = k + 1
        flip_img.save(new_img_path+data_dir+'/'+str(k)+'.png')
        k = k + 1
        label.append(steer_ang)
        label.append(-1*steer_ang)
        
    print(k)
    csv_file = new_csv_path+data_dir+'.csv'
    np.savetxt(csv_file, label, delimiter=",")        
            

