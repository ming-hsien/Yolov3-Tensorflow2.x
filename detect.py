import os
import cv2
import numpy as np
import tensorflow as tf
from yolov4.utils import detect_image, Load_Yolo_model
from yolov4.configs import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_path   = "./Data/test_image/test.jpg"

yolo = Load_Yolo_model()
detect_image(yolo, image_path, "./Data/test_image/results.jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
