from tensorflow import keras
import tensorflow as tf
import cv2
import time
from datetime import datetime
from PyQt5 import uic

from PyQt5 import QtCore, QtGui, QtWidgets
import threading
import imutils

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import math
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse


prototxtPath = os.path.abspath("SmartCameraModel/face_detector/deploy.prototxt")
weightsPath = os.path.abspath("SmartCameraModel/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
MaskModelTFlitePath = os.path.abspath("SmartCameraModel/model.tflite")
EncodingDataPath = os.path.abspath("SmartCameraModel/encodings.pickle")
CameraUIPath = os.path.abspath("SmartCameraView/camera_alarm.ui")
IconPath = os.path.abspath("SmartCameraView/")
FaceImagePath = os.path.abspath("SmartCameraModel/")