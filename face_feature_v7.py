#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow import keras
import tensorflow as tf
import math
import cv2
import imutils
import numpy as np
import SmartCamera_ShareMemory as sc_share_memory

from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (20, 20)

def detect_and_predict_mask(frame):
    global faceNet, maskNet
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            sc_share_memory.human_appear_status = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            sc_share_memory.face_area = (endX - startX) * (endY - startY)
            if (endX - startX) * (endY - startY) > 3000:
                sc_share_memory.face_detect_status = True
            else:
                sc_share_memory.face_detect_status = False

            locs.append((startX, startY, endX, endY))
    if not locs:
        sc_share_memory.face_detect_status = False
        sc_share_memory.human_appear_status = False
        
    if sc_share_memory.face_detect_status:
        sc_share_memory.global_locs = locs

    if len(faces) > 0:
        try:
            with session.as_default():
                with session.graph.as_default():
                    faces = np.array(faces, dtype="float32")
                    maskNet.set_tensor(input_details[0]['index'], faces)
                    maskNet.invoke()
                    preds = maskNet.get_tensor(output_details[0]['index'])
                    try:
                        if len(preds[0]) > 1:
                            if preds[0][1] > 0.5:
                                print(preds)
                                sc_share_memory.mask_detect_status = False
                            elif preds[0][0] > 0.5:
                                sc_share_memory.mask_detect_status = True
                    except Exception as exp:
                        print(str(exp))

        except Exception as exp:
            print(exp)
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"


faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
session = tf.Session(config=config)

keras.backend.set_session(session)

maskNet = tf.lite.Interpreter(model_path="model.tflite")
maskNet.allocate_tensors()

input_details = maskNet.get_input_details()
output_details = maskNet.get_output_details()
input_shape = input_details[0]['shape']

def calculate_average_temp(image):
    count = 0
    sum = 0

    for (x, y, window) in sliding_window(image, stepSize=5, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        temp = np.mean(image[y:y + winH, x:x + winW])
        if math.log10(temp) * 16.1 > 35:
            count += 1
            sum += math.log10(temp) * 16.1
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    if count > 0:
        return round(sum / count, 1)
    else:
        return 36.5
    
import time
def detect_face_version_7(frame, thermal):
    locs = sc_share_memory.global_locs
    #print(locs, sc_share_memory.face_detect_status, sc_share_memory.mask_detect_status)
    if locs:
        bbox = locs[0]
        
        face = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2])]
        cv2.imwrite("faces/" + str(datetime.now().strftime("%Y%m%d")) + ".jpg", face)
        
        temperature = calculate_average_temp(thermal[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        sc_share_memory.thermal_data = temperature
    else:
        sc_share_memory.face_detect_status = False
    time.sleep(0.1)
    
