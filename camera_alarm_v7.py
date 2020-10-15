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
import face_feature_v7
import SmartCamera_ShareMemory as sc_share_memory
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

frame = None
thermal = None

frame_face = {}
frame_face["frame"] = frame
frame_face["thermal"] = thermal

def grab_image(cam1, cam2):
    global frame_face
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

        frame_face["frame"] = frame
        frame_face["thermal"] = thermal



def grab_face_in():
    global face_detect_return
    while running:
        if frame_face["frame"] is not None and frame_face["thermal"] is not None:
            thermal = frame_face["thermal"]
            frame = frame_face["frame"]
            face_feature_v7.detect_face_version_7(frame, thermal)

def detect_face_and_mask():
    global face_detect_return
    while running:
        if frame_face["frame"] is not None and frame_face["thermal"] is not None:
            frame = frame_face["frame"]
            face_feature_v7.detect_and_predict_mask(frame)


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
import time

class Window(QtWidgets.QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = "THERMAL CAMERA"
        self.InitValue()
        self.InitUI()
        face_capture_thread.start()
        camera_capture_thread.start()
        detect_face_and_mask_capture_thread.start()
        self.show()

    def InitValue(self):
        global running
        running = True

    def InitUI(self):
        self.setWindowTitle(self.title)

        self.yes_ticker = QtGui.QPixmap("icon/NO-removebg-preview.png")
        self.no_ticker = QtGui.QPixmap("icon/YES-removebg-preview.png")
        self.no_mask = QtGui.QPixmap("icon/NOMASK-removebg-preview.png")
        self.mask = QtGui.QPixmap("icon/MASK-removebg-preview.png")
        self.fever = QtGui.QPixmap("icon/fever-removebg-preview.png")
        self.normal = QtGui.QPixmap("icon/normal-removebg-preview.png")
        self.position = QtGui.QPixmap("icon/position-removebg-preview.png")

        self.camera = OwnImageWidget(self.camera)

        self.ticker = self.ticker_label
        self.ticker.setPixmap(self.no_ticker)
        self.ticker_2 = self.ticker_label_2
        self.ticker_2.setPixmap(self.no_ticker)
        self.ticker_3 = self.ticker_label_3
        self.ticker_3.setPixmap(self.no_ticker)

        self.label = self.label
        self.label.setPixmap(self.position)
        self.label_2 = self.label_2
        self.label_2.setPixmap(self.normal)
        self.label_3 = self.label_3
        self.label_3.setPixmap(self.no_mask)

        fnt = self.tableWidget.font()
        fnt.setPointSize(30)
        self.tableWidget.setFont(fnt)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.face_update_frame)
        self.timer.start(0)

        self.timer_2 = QtCore.QTimer(self)
        self.timer_2.timeout.connect(self.data_update)
        self.timer_2.start(1000)

    def data_update(self):
        global face_detect_return
        if sc_share_memory.face_detect_status:
            self.ticker.setPixmap(self.yes_ticker)
        else:
            self.label.setPixmap(self.position)
            self.label_2.setPixmap(self.normal)
            self.label_3.setPixmap(self.no_mask)

            self.ticker.setPixmap(self.no_ticker)
            self.ticker_2.setPixmap(self.no_ticker)
            self.ticker_3.setPixmap(self.no_ticker)

        self.updateTable(sc_share_memory.thermal_data)
        if sc_share_memory.face_detect_status:
            if sc_share_memory.thermal_data > 38:
                self.label_2.setPixmap(self.fever)
                self.ticker_2.setPixmap(self.no_ticker)
            else:
                self.label_2.setPixmap(self.normal)
                self.ticker_2.setPixmap(self.yes_ticker)

        if sc_share_memory.mask_detect_status:
            self.label_3.setPixmap(self.mask)
            self.ticker_3.setPixmap(self.yes_ticker)
        else:
            self.label_3.setPixmap(self.no_mask)
            self.ticker_3.setPixmap(self.no_ticker)



    def face_update_frame(self):
        global running, frame_face
        if frame_face["frame"] is not None and frame_face["thermal"] is not None:
            frame = frame_face["frame"]
            thermal = frame_face["thermal"]
            frame = cv2.resize(frame, (591, 441))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            thermal = cv2.resize(thermal, (591, 441))
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            bpl = frame.shape[2] * frame.shape[1]

            image_in = QtGui.QImage(frame.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            thermal_in = QtGui.QImage(thermal.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            self.camera.setImage(image_in)
            #self.thermal.setImage(thermal_in)

    def updateTable(self, data):
        path = "faces/" + datetime.now().strftime("%Y%m%d") + ".jpg"
        if os.path.exists(path) and sc_share_memory.face_detect_status:
            
            rowposition = 0
            self.tableWidget.insertRow(rowposition)
            pic = QtGui.QPixmap(path)
            self.face_image = QtWidgets.QLabel(self.centralwidget)
            self.face_image.setScaledContents(True)
            self.face_image.setPixmap(pic)
            self.tableWidget.setCellWidget(rowposition, 0, self.face_image)

            # We have to divide this apart from self.tableWidget.setItem(rowposition, 1, QtWidgets.QTableWidgetItem(str(
            # sc_share_memory.thermal_data)).setTextAlignment(QtCore.Qt.AlignCenter))
            item = QtWidgets.QTableWidgetItem(str(sc_share_memory.thermal_data))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(rowposition, 1, item)

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
    face_capture_thread = threading.Thread(target=grab_face_in)
    camera_capture_thread = threading.Thread(target=grab_image, args=[-1, 2,])
    detect_face_and_mask_capture_thread = threading.Thread(target=detect_face_and_mask)
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())