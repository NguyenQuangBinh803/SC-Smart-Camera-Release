#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import queue
import sys
import threading
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import face_feature_v4

api_alarm = "http://192.168.1.184/control"

thresh = 49
running = False
form_class = uic.loadUiType("camera_alarm_v1.ui")[0]
q1 = queue.Queue()
q2 = queue.Queue()
face_capture_thread = None

face_detect_return = False

if not os.path.exists("faces"):
    os.makedirs("faces")

def grab_face_in(cam1, cam2, queue):
    global face_detect_return
    capture1 = cv2.VideoCapture(cam1)
    capture2 = cv2.VideoCapture(cam2)

    
    while (not capture1.isOpened() or not capture2.isOpened()):
        print("Try to reconnect camera face in ...")
        capture1 = cv2.VideoCapture(cam1)
        capture2 = cv2.VideoCapture(cam2)
        
    while (running):
        frame_face = {}
        human_data = {}
        bboxes_track = []

        retval2, thermal = capture2.read()
        retval1, frame = capture1.read()

        frame = frame[101: 381, 173: 560]
        thermal = cv2.resize(thermal, (387, 289))

        #threading.Thread(target=face_feature_v2.detect_faces, args=[frame, thermal, ]).start()

        face_detect_return, mask, temperature = face_feature_v4.detect_faces(frame, thermal)
        
        if q2.qsize() < 10 and face_detect_return:
            human_data["temperature"] = temperature
            human_data["mask"] = mask
            q2.put(human_data)

        frame_face["frame"] = frame
        frame_face["thermal"] = thermal

        if queue.qsize() < 3:
            queue.put(frame_face)
        if queue.qsize() > 2:
            print("show face in overload: ", queue.qsize())


class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class Window(QtWidgets.QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = "THERMAL CAMERA"
        self.InitValue()
        self.InitUI()
        face_capture_thread.start()
        self.show()

    def InitValue(self):
        global running
        running = True

    def InitUI(self):
        # set title text
        self.setWindowTitle(self.title)

        self.yes_ticker = QtGui.QPixmap("NO.png")
        self.no_ticker = QtGui.QPixmap("YES.png")

        # Turn in group box
        self.camera = OwnImageWidget(self.camera)
        #self.thermal = OwnImageWidget(self.thermal)
        self.ticker = self.ticker_label
        self.annoucement = self.label

        self.temp.setText(str(thresh))
        self.alarm_button.clicked.connect(self.change_thesh)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.face_update_frame)
        self.timer.start(0)

    def face_update_frame(self):
        global running, face_detect_return
        if not q1.empty():
            camera = q1.get()
            frame_face = camera["frame"]
            thermal = camera["thermal"]
            frame_face = cv2.resize(frame_face, (591, 441))
            frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

            thermal = cv2.resize(thermal, (591, 441))
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            bpl = frame_face.shape[2] * frame_face.shape[1]

            image_in = QtGui.QImage(frame_face.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            thermal_in = QtGui.QImage(thermal.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            self.camera.setImage(image_in)
            #self.thermal.setImage(thermal_in)
            if face_detect_return:
                self.ticker.setPixmap(self.yes_ticker)
            else:
                self.ticker.setPixmap(self.no_ticker)

            if not q2.empty():
                data_temp = q2.get()
                temperature_data = [data_temp["temperature"]]
                mask_data = data_temp["mask"]
                if mask_data:
                    
                    self.annoucement.setText("With Mask")
                    self.annoucement.setStyleSheet('color: green')
                else:
                    self.annoucement.setText("No Mask")
                    self.annoucement.setStyleSheet('color: red')
                self.updateTable(temperature_data)

    def updateTable(self, data):
        self.ticker.setPixmap(self.yes_ticker)
        rowposition = 0
        self.tableWidget.insertRow(rowposition)
        path = "faces/" + datetime.now().strftime("%Y%m%d") + ".jpg"
        if os.path.exists(path):
            pic = QtGui.QPixmap(path)
            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setScaledContents(True)
            self.label.setPixmap(pic)
            self.tableWidget.setCellWidget(rowposition, 0, self.label)
            self.tableWidget.setItem(rowposition, 1, QtWidgets.QTableWidgetItem(str(data[-1])))

        self.tableWidget.verticalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # kq = requests.post(api_alarm, data=data_control).json()
        # print(kq)

    def change_thesh(self):
        global thresh
        thresh = int(self.temp.text())

    def closeEvent(self, event):
        global running
        running = False
        event.accept()

if __name__ == '__main__':
    face_capture_thread = threading.Thread(target=grab_face_in, args=(-1, 2, q1))
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())
