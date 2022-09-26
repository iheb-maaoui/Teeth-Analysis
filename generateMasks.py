
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import json

def color_teeth(img,all_x,all_y):
  contours=[]
  for i in range(len(all_x)):
    contours.append([all_x[i],all_y[i]])
  
  cv2.fillPoly(img, pts = [np.array(contours)], color =(255,255,255))
  return img

def color_background(img):
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      img[i,j,:] = np.array([0.0,0.0,0.0])
  return img.astype(np.int32)

def generate_mask(file_path):
  import json

  img = cv2.imread(file_path+'.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_msk = img.copy()
  img_msk = color_background(img_msk)
  with open(file_path+'.json', 'r') as f:
    data = json.load(f)
    teeths = data['polygon']

    for j in range(len(teeths)):
      all_x = teeths[j]['coords']['all_x']
      all_y = teeths[j]['coords']['all_y']
      for i in range(len(all_x)):
        color_teeth(img_msk,all_x,all_y)
  return img_msk

import os
for i in range(1,52):
  if(i!=29):
    file_path = f'/home/iheb/teethAnalysis/images/img_{i}_4'
    img = generate_mask(file_path)
    cv2.imwrite(f'/home/iheb/masks/img_{i}_4'+'_msk.png', img)