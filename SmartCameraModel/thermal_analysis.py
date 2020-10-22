#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import smart_camera_share_memory as sc_share_memory
from smart_camera_common_imports import *

class ThermalAnalysis:
    def __init__(self):
        pass


    def pyramid(self, image, scale=1.5, minSize=(30, 30)):
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            yield image

    def sliding_window(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    def calculate_average_temp(self, image):
        count = 0
        sum = 0
        (winW, winH) = (10, 10)
        for (x, y, window) in self.sliding_window(image, stepSize=5, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            temp = np.mean(image[y:y + winH, x:x + winW])
            if math.log10(temp) * 16 > 35:
                count += 1
                sum += math.log10(temp) * 16
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        if count > 0:
            return round(sum / count, 1)
        else:
            return 36.5


    def calculate_thermal(self, thermal):
        locs = sc_share_memory.global_locs
        if locs:
            bbox = locs[0]
            temperature = self.calculate_average_temp(thermal[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
            sc_share_memory.thermal_data = temperature
        else:
            sc_share_memory.face_detect_status = False
