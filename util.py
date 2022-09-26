
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import json

def load_data(images_path,masks_path):
  X=[]
  for _,_,filenames in os.walk(images_path):
    for filename in sorted(filenames):
      if(filename.split('.')[-1]=='png'):
        img = cv2.imread(os.path.join(images_path,filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img/255.0
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_NEAREST)
        X.append(img.astype(np.float32))
  
  y = []
  for _,_,filenames in os.walk(masks_path):
    for filename in sorted(filenames):

      if(filename.split('.')[-1]=='png'):
        img = cv2.imread(os.path.join(masks_path,filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img/255.0
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_NEAREST)
        y.append(img.astype(np.float32))
  
  return (np.array(X),np.array(y))

  