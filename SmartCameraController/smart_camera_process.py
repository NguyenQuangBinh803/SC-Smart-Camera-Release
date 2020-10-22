#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smart_camera_share_memory as sc_share_memory

import math
import cv2
import imutils
import numpy as np

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


(winW, winH) = (10, 10)

def detect_and_predict_mask(frame):
    (h, w) = frame.shape[:2]
    print(1)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    print(1)
    sc_share_memory.faceNet.setInput(blob)
    print(1)
    detections = sc_share_memory.faceNet.forward()
    print(1)
    faces = []
    locs = []
    preds = []
    print(1)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            sc_share_memory.human_appear_status = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            cv2.imwrite("faces/" + str(datetime.now().strftime("%Y%m%d")) + ".jpg", face)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            sc_share_memory.global_face_image = faces
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
            with sc_share_memory.session.as_default():
                with sc_share_memory.session.graph.as_default():
                    faces = np.array(faces, dtype="float32")
                    sc_share_memory.maskNet.set_tensor(sc_share_memory.input_details[0]['index'], faces)
                    sc_share_memory.maskNet.invoke()
                    preds = sc_share_memory.maskNet.get_tensor(sc_share_memory.output_details[0]['index'])
                    try:
                        if len(preds[0]) > 1:
                            if preds[0][1] > 0.5:
                                # print(preds)
                                sc_share_memory.mask_detect_status = False
                            elif preds[0][0] > 0.5:
                                sc_share_memory.mask_detect_status = True
                    except Exception as exp:
                        print(str(exp))

        except Exception as exp:
            print(exp)
    return (locs, preds)

def calculate_average_temp(image):
    count = 0
    sum = 0

    for (x, y, window) in sliding_window(image, stepSize=5, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        temp = np.mean(image[y:y + winH, x:x + winW])
        if math.log10(temp) * 16.1 > 35:
            count += 1
            # print(math.log10(temp) * 16.1, temp)
            sum += math.log10(temp) * 16.1
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    if count > 0:
        return round(sum / count, 1)
    else:
        return 36.5


import time


def calculate_thermal(frame, thermal):
    locs = sc_share_memory.global_locs
    # print(locs, sc_share_memory.face_detect_status, sc_share_memory.mask_detect_status)
    if locs:
        bbox = locs[0]
        temperature = calculate_average_temp(thermal[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        sc_share_memory.thermal_data = temperature
    else:
        sc_share_memory.face_detect_status = False
    time.sleep(0.1)


running = True

def grab_image(cam1, cam2):
    capture1 = cv2.VideoCapture(cam1)
    capture2 = cv2.VideoCapture(cam2)

    while (not capture1.isOpened() or not capture2.isOpened()):
        print("Try to reconnect camera face in ...")
        capture1 = cv2.VideoCapture(cam1)
        capture2 = cv2.VideoCapture(cam2)

    while (running):

        retval2, thermal = capture2.read()
        retval1, frame = capture1.read()

        frame = frame[101: 381, 173: 560]
        thermal = cv2.resize(thermal, (387, 289))

        sc_share_memory.frame_face["frame"] = frame
        sc_share_memory.frame_face["thermal"] = thermal


def grab_face_in():
    print("Start calculation")
    while running:
        print("Start calculation")
        if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
            thermal = sc_share_memory.frame_face["thermal"]
            frame = sc_share_memory.frame_face["frame"]
            calculate_thermal(frame, thermal)

def detect_face_and_mask():
    print("Start detection")
    while running:
        print("Start detection")
        if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
            print("Start detection")
            frame = sc_share_memory.frame_face["frame"]
            detect_and_predict_mask(frame)

