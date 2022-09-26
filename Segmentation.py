
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
from util import load_data
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
import cv2
from keras import backend as keras
from model import myUnet
from losses import *

images_path = '/home/iheb/teethAnalysis/images'
masks_path = '/home/iheb/masks'

(X,y) = load_data(images_path,masks_path)

y = np.expand_dims(y,axis=-1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
x_train, x_test, y_train, y_test = np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

def myGenerator(train_generator,mask_generator):
  gen = zip(train_generator,mask_generator)

  for (img,label) in gen:
    yield (img,label)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_data_gen_args = dict(rotation_range=90,
                     vertical_flip=True)
mask_data_gen_args = dict(rotation_range=90,
                     vertical_flip=True)
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow(x_train, batch_size=1,seed=42)
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(y_train, batch_size=1,seed=42)
gen = myGenerator(image_generator,mask_generator)


model = myUnet()

model.fit(gen,epochs=10,batch_size = 1,shuffle=True,steps_per_epoch=150)

model.evaluate(x_test,y_test)

ypred = model.predict(x_test)

x_test.shape

model.save('model_bce.h5')

model_weighted = myUnet()
model_weighted = model_weighted.get_unet()

model_weighted.fit(gen,epochs=10,batch_size = 1,shuffle=True,steps_per_epoch=150)

ypred_weighted_bce = model_weighted.predict(x_test)

model_focal = myUnet()
model_focal = model_focal.get_unet()

model_focal.fit(gen,epochs=10,batch_size = 1,shuffle=True,steps_per_epoch=150)

ypred_focal = model_focal.predict(x_test)

model_weighted_dice = myUnet()
model_weighted_dice = model_weighted_dice.get_unet()

model_weighted_dice.fit(gen,epochs=10,batch_size = 1,shuffle=True,steps_per_epoch=150)

ypred_weighted_dice1 = model_weighted_dice.predict(x_test)

model_weighted_dice = myUnet()
model_weighted_dice = model_weighted_dice.get_unet()

model_weighted_dice.fit(gen,epochs=10,batch_size = 1,shuffle=True,steps_per_epoch=150)

ypred_weighted_dice2 = model_weighted_dice.predict(x_test)

model_weighted_dice = myUnet()
model_weighted_dice = model_weighted_dice.get_unet()

fig, axs = plt.subplots(5,6,figsize=(20, 20))
for i in range(len(x_test)):
  axs[i,0].imshow(y_test[i][...,0])
  axs[i,1].imshow(ypred[i][...,0])
  axs[i,2].imshow(ypred_weighted_bce[i][...,0])
  axs[i,3].imshow(ypred_focal[i][...,0])
  axs[i,4].imshow(ypred_weighted_dice1[i][...,0])
  axs[i,5].imshow(ypred_weighted_dice2[i][...,0])

