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


class Window(QtWidgets.QMainWindow, uic.loadUiType(CameraUIPath)[0]):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = "SMART CAMERA SYSTEM"
        self.InitUI()
        self.show()

    def InitUI(self):
        self.setWindowTitle(self.title)
        print(IconPath + "/icon/NO-removebg-preview.png")

        self.yes_ticker = QtGui.QPixmap(IconPath + "/icon/NO-removebg-preview.png")
        self.no_ticker = QtGui.QPixmap(IconPath + "/icon/YES-removebg-preview.png")
        self.no_mask = QtGui.QPixmap(IconPath + "/icon/NOMASK-removebg-preview.png")
        self.mask = QtGui.QPixmap(IconPath + "/icon/MASK-removebg-preview.png")
        self.fever = QtGui.QPixmap(IconPath + "/icon/fever-removebg-preview.png")
        self.normal = QtGui.QPixmap(IconPath + "/icon/normal-removebg-preview.png")
        self.position = QtGui.QPixmap(IconPath + "/icon/position-removebg-preview.png")
        self.move_forward = QtGui.QPixmap(IconPath + "/icon/forward.png")

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

        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)


        self.face_image = self.label_5
        self.face_image.setScaledContents(True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.face_update_frame)
        self.timer.start(0)

        self.timer_2 = QtCore.QTimer(self)
        self.timer_2.timeout.connect(self.data_update)
        self.timer_2.start(0)

    def data_update(self):
        print(sc_share_memory.face_area)
        if sc_share_memory.human_appear_status:
            if sc_share_memory.face_area > 3000:
                self.label.setPixmap(self.position)
                self.ticker.setPixmap(self.yes_ticker)
            elif sc_share_memory.face_area < 3000:
                self.label.setPixmap(self.move_forward)
                self.label_2.setPixmap(self.normal)
                self.label_3.setPixmap(self.no_mask)

                self.ticker.setPixmap(self.no_ticker)
                self.ticker_2.setPixmap(self.no_ticker)
                self.ticker_3.setPixmap(self.no_ticker)
                self.label_4.setText("")
        else:
            self.label.setPixmap(self.position)
            self.label_2.setPixmap(self.normal)
            self.label_3.setPixmap(self.no_mask)

            self.ticker.setPixmap(self.no_ticker)
            self.ticker_2.setPixmap(self.no_ticker)
            self.ticker_3.setPixmap(self.no_ticker)
            self.label_4.setText("")
            self.label_6.setText("")

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

        if sc_share_memory.frame_face["frame"] is not None and sc_share_memory.frame_face["thermal"] is not None:
            frame = sc_share_memory.frame_face["frame"]
            thermal = sc_share_memory.frame_face["thermal"]
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
        path = FaceImagePath + "/faces/" + datetime.now().strftime("%Y%m%d") + ".jpg"
        # and sc_share_memory.face_detect_status
        if os.path.exists(path):
            pic = QtGui.QPixmap(path)
            self.face_image.setPixmap(pic)
            if sc_share_memory.human_appear_status: 
                if sc_share_memory.face_detect_status:
                    self.label_4.setText(str(sc_share_memory.thermal_data))
                if sc_share_memory.global_human_name:
                    self.label_6.setText(str(sc_share_memory.global_human_name))

    def change_thesh(self):
        global thresh
        thresh = int(self.temp.text())

    def closeEvent(self, event):
        global running
        running = False
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())
