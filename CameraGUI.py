#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import sys
import os
import threading
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtGui, QtWidgets, uic

form_class = uic.loadUiType("camera_alarm_v1.ui")[0]

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

thresh = 49
class Window(QtWidgets.QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = "THERMAL CAMERA"
        self.InitValue()
        self.InitUI()
        # face_capture_thread.start()
        self.show()

    def InitValue(self):
        global running
        running = True

    def InitUI(self):
        # set title text
        self.setWindowTitle(self.title)

        # Turn in group box
        self.camera = OwnImageWidget(self.camera)
        self.thermal = OwnImageWidget(self.thermal)

        self.temp.setText(str(thresh))
        self.alarm_button.clicked.connect(self.change_thesh)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.face_update_frame)
        self.timer.start(0)

    def face_update_frame(self):

        global running

        frame_face = np.zeros((500,500,3), np.uint8)
        thermal = np.zeros((500,500,3), np.uint8)

        bpl = frame_face.shape[2] * frame_face.shape[1]

        image_in = QtGui.QImage(frame_face.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
        thermal_in = QtGui.QImage(thermal.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)

        self.camera.setImage(image_in)
        self.thermal.setImage(thermal_in)

        # if not q1.empty():
        #     camera = q1.get()
        #     frame_face = camera["frame"]
        #     thermal = camera["thermal"]
        #     frame_face = cv2.resize(frame_face, (591, 441))
        #     frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        #
        #     thermal = cv2.resize(thermal, (591, 441))
        #     thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        #     bpl = frame_face.shape[2] * frame_face.shape[1]
        #
        #     image_in = QtGui.QImage(frame_face.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
        #     thermal_in = QtGui.QImage(thermal.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
        #     self.camera.setImage(image_in)
        #     self.thermal.setImage(thermal_in)
        #
        #     if not q2.empty():
        #         data_temp = q2.get()
        #         self.updateTable(data_temp)

    def updateTable(self, data):
        rowposition = self.tableWidget.rowCount()
        self.tableWidget.insertRow(rowposition)
        for column_number, column_data in enumerate(data):
            item = str(column_data)
            if column_number == 0:
                path = "faces/" + item + ".jpg"
                if os.path.exists(path):
                    pic = QtGui.QPixmap(path)
                    self.label = QtWidgets.QLabel(self.centralwidget)
                    self.label.setScaledContents(True)
                    self.label.setPixmap(pic)
                    self.tableWidget.setCellWidget(rowposition, column_number, self.label)
                else:
                    self.tableWidget.setItem(rowposition, column_number, QtWidgets.QTableWidgetItem("Missing"))
            else:
                self.tableWidget.setItem(rowposition, column_number, QtWidgets.QTableWidgetItem(item))
        self.tableWidget.verticalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        if data[1] > thresh:
            data_control = '{"PASSWORD": "PYROJECTCOLTD","CMD": "STOP"}'
        else:
            data_control = '{"PASSWORD": "PYROJECTCOLTD","CMD": "ALLOW"}'
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
    # face_capture_thread = threading.Thread(target=grab_face_in, args=(-1, 2, q1))

    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()

    sys.exit(app.exec_())