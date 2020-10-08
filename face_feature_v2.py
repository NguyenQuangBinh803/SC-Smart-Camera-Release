# import the necessary packages
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
import math

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
            print(locs)

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
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

def detect_faces(frame, thermal):
    (locs, _) = detect_and_predict_mask(frame, faceNet, maskNet)
    for box in locs:
        bbox = box
        size = 15
        offset = 0
        # boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.rectangle(frame, (int((bbox[0] + bbox[2]) / 2 - size/2), int(bbox[3] + offset)),
                      (int((bbox[0] + bbox[2]) / 2 + size/2), int(bbox[3] + size + offset)), (0, 0, 255), 1)
        # print("Face area: ", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        TEST_RESULT = "logging/" + datetime.now().strftime("%Y%m%d_%H%M%S%f")
        cv2.imwrite(TEST_RESULT + ".png", (thermal[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]))

        area = thermal[int(bbox[3] + offset):int(bbox[3] + size + offset),
               int(((bbox[0] + bbox[2]) / 2) - size):int(((bbox[0] + bbox[2]) / 2) + size)]

        TEST_RESULT = "ROI/" + datetime.now().strftime("%Y%m%d_%H%M%S%f")
        cv2.imwrite(TEST_RESULT + ".png", area)
        temp = np.mean(area)

        temperature = round((math.log10(temp) * 16), 1)
        print("Mean pixel value:", temp, temperature)
        cv2.putText(frame, str(temperature), (int(bbox[0]), int(bbox[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)
