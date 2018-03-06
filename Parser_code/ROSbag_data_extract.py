#!/usr/bin/env python


import sys
import numpy as np
import cv2
import rospy
import rosbag
import csv
import os


path_name = '/home/dkr92/catkin_ws/src/udacity_datset/scripts/Train_bags/'
file_name = 'HMB_6.bag'
bag = rosbag.Bag(path_name+file_name)

topic_img = '/center_camera/image_color/compressed'
topic_steer = '/vehicle/steering_report'

img_dir = '/home/dkr92/data/Set_6/'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

resolution = (256,192) #Resize Resolution

flag = 0
cnt = 0

steer_ang_arr = []

for topic, msg, t in bag.read_messages():
    
    if topic == topic_img:
        
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, resolution)
        
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:         
            cv2.destroyAllWindows()
            
        img_file = img_dir+str(cnt)+'.png'
        cv2.imwrite(img_file,img)
        flag = 1
        
    if topic == topic_steer and flag == 1:
        print msg.steering_wheel_angle
        steer_ang_arr.append(msg.steering_wheel_angle)
        flag = 0
        cnt += 1
        print ('')

steer_ang_arr = np.array(steer_ang_arr)
csv_file = '/home/dkr92/data/steering_report_6.csv'
np.savetxt(csv_file, steer_ang_arr, delimiter=",")

print ("img_cnt:", cnt)
print ("steer_cnt:", cnt)
cv2.destroyAllWindows()