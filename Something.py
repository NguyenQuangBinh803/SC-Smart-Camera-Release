import os
import sys

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication
# import face_feature_v3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        capture1 = cv2.VideoCapture(0)
        capture2 = cv2.VideoCapture(0)
        while True:
            ret, frame = capture1.read()
            # ret, thermal = capture2.read()
            thermal = np.zeros((640, 480, 3), np.uint8)

            if ret:
                frame = frame[101: 381, 173: 560]
                thermal = cv2.resize(thermal, (387, 289))
                face_feature_v3.detect_faces(frame, thermal)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thremal_image = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets, uic
form_class = uic.loadUiType("camera_alarm_v4.ui")[0]


class App(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.title = ''
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(self.no_ticker)
    #
    def initUI(self):

        self.yes_ticker = QtGui.QPixmap("NO.png")
        self.no_ticker = QtGui.QPixmap("YES.png")
        self.setWindowTitle(self.title)
        # th = Thread(self)
        self.label.setPixmap(self.yes_ticker)

        # th.changePixmap.connect(self.setImage)
        # th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
