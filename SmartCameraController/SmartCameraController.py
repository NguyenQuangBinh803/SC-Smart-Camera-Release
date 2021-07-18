#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SmartCameraCommonImports import *
from SmartCameraModel.ThermalAnalysis import ThermalAnalysis
from SmartCameraModel.FaceDetectAndRecognition import FaceDetectAndRecognition
import SmartCameraShareMemory as sc_share_memory


class SmartCameraController:
    def __init__(self):
        self.running = True
        self.thermal_analysis = ThermalAnalysis()
        self.face_detect_recognize = FaceDetectAndRecognition()

    def threading_streaming(self, camera_1, camera_2):

        normal_camera = cv2.VideoCapture(camera_1)
        thermal_camera = cv2.VideoCapture(camera_2)
        
        while (not normal_camera.isOpened() or not thermal_camera.isOpened()):
            normal_camera = cv2.VideoCapture(camera_1)
            thermal_camera = cv2.VideoCapture(camera_2)

        while (self.running):
            retval2, thermal = thermal_camera.read()
            retval1, frame = normal_camera.read()
            

            frame = frame[101: 381, 173: 560]
            thermal = cv2.resize(thermal, (387, 289))

            sc_share_memory.frame_face["frame"] = frame
            sc_share_memory.frame_face["thermal"] = thermal
            time.sleep(0.02)

    def threading_calculate_temperature(self):
        while (self.running):
            

            # This thermal calculation is only use global_locs of the face and the thermal_frame to estimate the temperature
            # This is also unclean that the inside and outside condition is supposed to be one

            if sc_share_memory.frame_face["thermal"] is not None:
                print("Start calculate temperature ... ")
                thermal = sc_share_memory.frame_face["thermal"]
                self.thermal_analysis.calculate_thermal(thermal)
            time.sleep(0.05)

    def threading_detect_face(self):
        while (self.running):
            print("Start detecting face ... ")
            if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
                frame = sc_share_memory.frame_face["frame"]

                # This solution suppose to reduce one thread for the system, but in contrast, make the model not plain,
                #  clear and reusable. So how to solve this with making 1 thread and remain the model plain and clear ????
                # names = self.face_detect_recognize.face_detect_and_mask(frame)
                # print(names)
                # sc_share_memory.global_human_name = names
                self.face_detect_recognize.face_detect_and_mask(frame)
                # if len(names) > 0:
                #     sc_share_memory.global_human_name = names[0]
                #     print(names)
            time.sleep(0.05)

    def threading_detect_mask(self):
        while (self.running):
            
            if sc_share_memory.global_face_image is not None:
                print("Start detecting mask ... ")
                sc_share_memory.global_human_name = self.face_detect_recognize.face_recognize_openface(sc_share_memory.global_face_image)
            time.sleep(0.05)


if __name__ == "__main__":
    smart_camera_controller = SmartCameraController()
    threading.Thread(target=smart_camera_controller.threading_streaming, args=[-1, 2, ]).start()
    threading.Thread(target=smart_camera_controller.threading_detect_mask).start()
    threading.Thread(target=smart_camera_controller.threading_detect_face).start()
    while True:

        if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
            print("Frame receive")
            frame = sc_share_memory.frame_face["frame"]
            thermal = sc_share_memory.frame_face["thermal"]
            cv2.imshow("thermal", thermal)
            cv2.imshow("frame", frame)
        else:
            print("No frame receive")
        cv2.waitKey(1)

