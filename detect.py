import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from yolov4.utils import detect_image, Load_Yolo_model
from yolov4.configs import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_dir = "./Data/test_image"
save_dir = "./Data/results/"
img_list = glob.glob(img_dir + '/*')

yolo = Load_Yolo_model()

# detect_image(yolo, image_path, "./Data/test_image/results.jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

for img_path in img_list:
    detect_image(yolo, image_path, os.path.join(save_dir,os.path.basename(img_path)), input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
