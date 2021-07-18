
import tensorflow as tf
import cv2
import time
import threading
import imutils
import math
import numpy as np
import pickle
import os

from datetime import datetime
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from glob import glob

Path_FaceDetectorProto = os.path.abspath("SmartCameraModel/MathematicalModel/FaceDetector/deploy.prototxt")
Path_FaceDetectorWeights = os.path.abspath("SmartCameraModel/MathematicalModel/FaceDetector/res10_300x300_ssd_iter_140000.caffemodel")

Path_MaskDetectorModel = os.path.abspath("SmartCameraModel/MathematicalModel/MaskDetector.tflite")

# Path_FaceRecognizerDlibEncodings = os.path.abspath("SmartCameraModel/MathematicalModel/encodings.pickle")
Path_FaceRecognizerFeatureExtractor = os.path.abspath("SmartCameraModel/MathematicalModel/FaceFeatureExtractor.t7")
Path_FaceRecognizerFeatureList = os.path.abspath("SmartCameraModel/MathematicalModel/FaceFeature.pickle")
Path_FaceRecognizerLabelList = os.path.abspath("SmartCameraModel/MathematicalModel/FaceLabel.pickle")

Path_FaceTemporaryImage = os.path.abspath("SmartCameraView/TemporaryFaceImage/")


Path_UI_Design = os.path.abspath("SmartCameraView/camera_alarm.ui")
Path_UI_Icon = os.path.abspath("SmartCameraView/")
Path_FaceImage = os.path.abspath("SmartCameraModel/") 