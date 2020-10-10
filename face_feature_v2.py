#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import cv2
import imutils
import numpy as np

from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
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


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))


    # only make a predictions if at least one face was detected
    # if len(faces) > 0:
    #     # for faster inference we'll make batch predictions on *all*
    #     # faces at the same time rather than one-by-one predictions
    #     # in the above `for` loop
    #     faces = np.array(faces, dtype="float32")
    #     preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")


def calculate_average_temp(image):
    count = 0
    sum = 0

    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=5, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        temp = np.mean(image[y:y + winH, x:x + winW])
        if math.log10(temp) * 16 > 35:
            count += 1
            sum += math.log10(temp) * 16
            # print(math.log10(temp) * 16)
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    if count > 0:
        return round(sum / count, 1)
    else:
        return 36.5


def detect_faces(frame, thermal):
    (locs, _) = detect_and_predict_mask(frame, faceNet, maskNet)
    temperature = 36.5
    ret = False
    for box in locs:
        bbox = box
        face = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2])]
        cv2.imwrite("faces/" + str(datetime.now().strftime("%Y%m%d")) + ".jpg", face)
        temperature = calculate_average_temp(thermal[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        print("Face area: ", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 4500:
            ret = True

    return ret, temperature
