import os
import pickle
import queue
import sys
import threading

import cv2
import imutils
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import face_feature_v2

api_alarm = "http://192.168.1.184/control"

# deep sort
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
id_detected = {}

thresh = 49
running = False
form_class = uic.loadUiType("camera_alarm_v1.ui")[0]
q1 = queue.Queue()
q2 = queue.Queue()
face_capture_thread = None

if not os.path.exists("faces"):
    os.makedirs("faces")


def grab_face_in(cam1, cam2, queue):
    capture1 = cv2.VideoCapture(cam1)
    capture2 = cv2.VideoCapture(cam2)
    while (not capture1.isOpened() or not capture2.isOpened()):
        print("Try to reconnect camera face in ...")
        capture1 = cv2.VideoCapture(cam1)
        capture2 = cv2.VideoCapture(cam2)
    while (running):
        frame_face = {}
        bboxes_track = []
        capture1.grab()
        capture2.grab()
        retval1, frame = capture1.retrieve(0)
        retval2, thermal = capture2.retrieve(0)

        frame = frame[101: 381, 173: 560]
        thermal = cv2.resize(thermal, (387, 289))
        # frame = imutils.resize(frame, width=400)
        # thermal = imutils.resize(thermal, width=400)

        face_feature_v2.detect_faces(frame, thermal)

        frame_face["frame"] = frame
        frame_face["thermal"] = thermal

        if queue.qsize() < 10:
            queue.put(frame_face)
        if queue.qsize() > 9:
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
        if not q1.empty():
            camera = q1.get()
            frame_face = camera["frame"]
            thermal = camera["thermal"]
            frame_face = cv2.resize(frame_face, (591, 441), interpolation=cv2.INTER_CUBIC)
            frame_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)

            thermal = cv2.resize(thermal, (591, 441), interpolation=cv2.INTER_CUBIC)
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            bpl = frame_face.shape[2] * frame_face.shape[1]

            image_in = QtGui.QImage(frame_face.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            thermal_in = QtGui.QImage(thermal.data, 591, 441, bpl, QtGui.QImage.Format_RGB888)
            self.camera.setImage(image_in)
            self.thermal.setImage(thermal_in)

            if not q2.empty():
                data_temp = q2.get()
                self.updateTable(data_temp)

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
    face_capture_thread = threading.Thread(target=grab_face_in, args=(1, 0, q1))
    app = QtWidgets.QApplication(sys.argv)
    main_window = Window()
    sys.exit(app.exec_())
