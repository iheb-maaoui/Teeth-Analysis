!pip install tensorflow==2.7.0

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

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

def generate_bbox(file_path):
  import json

  img = cv2.imread(file_path+'.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_msk = img.copy()
  img_msk = color_background(img_msk)
  with open(file_path+'.json', 'r') as f:
    data = json.load(f)
    teeths = data['polygon']
    bboxes = []
    for j in range(len(teeths)):
      all_x = teeths[j]['coords']['all_x']
      all_y = teeths[j]['coords']['all_y']

      xmin = min(all_x)
      xmax = max(all_x)

      ymin = min(all_y)
      ymax = max(all_y)

      bboxes.append([xmin,xmax,ymin,ymax])
      
  return bboxes

import os
import pandas as pd
data = pd.DataFrame(columns=['id','xmin','xmax','ymin','ymax','label'])
for i in range(1,52):
  if(i!=29):
    file_path = f'/home/iheb/teethAnalysis/img_{i}_4'
    file_name = f'img_{i}_4.png'
    bounding_boxes = generate_bbox(file_path)
    for bounding_box in bounding_boxes:
      image_features = []
      image_features.append(i)
      
      image_features+=[float(i) for i in bounding_box]
      label = 'teeth'
      image_features.append(label)
      data = data.append(pd.DataFrame([image_features], columns=['id','xmin','xmax','ymin','ymax','label']), ignore_index=True)

data

def plot_bbox(img_id,data):
  img = cv2.imread(os.path.join('/home/iheb/teethAnalysis',f'img_{img_id}_4.png'))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  height, width, channel = img.shape
  print(f"Image: {img.shape}")
  bboxs = data[data['id']==img_id]
  for index, row in bboxs.iterrows():
      xmin = row['xmin']
      xmax = row['xmax']
      ymin = row['ymin']
      ymax = row['ymax']
      xmin = int(xmin)
      xmax = int(xmax)
      ymin = int(ymin)
      ymax = int(ymax)
      label_name = row['label']
      print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
      cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 1)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, 'teeth', (xmin,ymin-10), font, 0.8, (0,255,0), 1)
  plt.figure(figsize=(15,10))
  plt.title('Image with Bounding Box')
  plt.imshow(img)
  plt.axis("off")
  plt.show()

plot_bbox(1,data)

images_ids = []
for image_id in os.listdir('/home/iheb/teethAnalysis'):
  if(image_id.split('.')[-1] == 'png'):
    images_ids.append(image_id)
train_images = images_ids[:45]
test_images = images_ids[45:]

pip install pascal-voc-writer

from pascal_voc_writer import Writer
for image_id in train_images:
  if(image_id.split('.')[-1]=='png'):

    writer = Writer(os.path.join('/home/iheb/teethAnalysis/',image_id), 799,533)

    bboxs = data[data['id']==int(image_id.split('.')[0].split('_')[-2])]
    for index, row in bboxs.iterrows():
      xmin = row['xmin']
      xmax = row['xmax']
      ymin = row['ymin']
      ymax = row['ymax']
      xmin = int(xmin)
      xmax = int(xmax)
      ymin = int(ymin)
      ymax = int(ymax)
      label_name = row['label']
      writer.addObject(label_name, xmin, ymin, xmax, ymax)
    writer.save(os.path.join('/home/iheb/trainingdemo/train',image_id.split('.')[0]+'.xml'))

import shutil

for image_id in train_images:
  shutil.copy(f'/home/iheb/teethAnalysis/{image_id}',f'/home/iheb/trainingdemo/train/{image_id}')

from pascal_voc_writer import Writer
for image_id in test_images:
  if(image_id.split('.')[-1]=='png'):

    writer = Writer(os.path.join('/home/iheb/teethAnalysis/',image_id), 799,533)

    bboxs = data[data['id']==int(image_id.split('.')[0].split('_')[-2])]
    for index, row in bboxs.iterrows():
      xmin = row['xmin']
      xmax = row['xmax']
      ymin = row['ymin']
      ymax = row['ymax']
      xmin = int(xmin)
      xmax = int(xmax)
      ymin = int(ymin)
      ymax = int(ymax)
      label_name = row['label']
      writer.addObject(label_name, xmin, ymin, xmax, ymax)
    writer.save(os.path.join('/home/iheb/trainingdemo/test',image_id.split('.')[0]+'.xml'))

for image_id in test_images:
  shutil.copy(f'/home/iheb/teethAnalysis/{image_id}',f'/home/iheb/trainingdemo/test/{image_id}')

!git clone https://github.com/tensorflow/models.git

cd /home/iheb/models/research

!protoc object_detection/protos/*.proto --python_out=.

!git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

!make

cp -r pycocotools /home/iheb/models/research

pwd

cd ..

cd ..

cp object_detection/packages/tf2/setup.py .

!python -m pip install .

!python object_detection/builders/model_builder_tf2_test.py

cd /home/iheb/trainingdemo/pre-trained-models

!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz

!tar -xvf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz

!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz

cd pre-trained-models

!tar -xvf ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz

cd /home/iheb/trainingdemo

!python generate_tfrecord.py -x /home/iheb/trainingdemo/train -l /home/iheb/trainingdemo/annotations/label_map.pbtxt -o /home/iheb/trainingdemo/annotations/train.record

!python generate_tfrecord.py -x /home/iheb/trainingdemo/test -l /home/iheb/trainingdemo/annotations/label_map.pbtxt -o /home/iheb/trainingdemo/annotations/test.record

cd /home/iheb/trainingdemo

!python model_main_tf2.py --model_dir=/home/iheb/trainingdemo/mymodels/fasterRcnn --pipeline_config_path=/home/iheb/trainingdemo/mymodels/fasterRcnn/pipeline.config

!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /home/iheb/trainingdemo/mymodels/fasterRcnn/pipeline.config --trained_checkpoint_dir /home/iheb/trainingdemo/mymodels/fasterRcnn --output_directory /home/iheb/trainingdemo/exported_models

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/home/iheb/trainingdemo/test/img_51_4.png'


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/home/iheb/trainingdemo/exported_models'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/iheb/trainingdemo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.5,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED

from google.colab import files
files.download("/home/iheb/trainingdemo/mymodels")

