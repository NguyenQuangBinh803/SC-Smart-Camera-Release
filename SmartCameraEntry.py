#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SmartCameraView.SmartCameraUi import *
from SmartCameraController.SmartCameraController import SmartCameraController

if __name__ == "__main__":
    smart_camera_controller = SmartCameraController()
    app = QtWidgets.QApplication(sys.argv)

    threading.Thread(target=smart_camera_controller.threading_streaming, args=[-1, 2, ]).start()
    threading.Thread(target=smart_camera_controller.threading_detect_mask).start()
    threading.Thread(target=smart_camera_controller.threading_detect_face).start()
    threading.Thread(target=smart_camera_controller.threading_calculate_temperature).start()

    main_window = Window()
    sys.exit(app.exec_())


def training_learning_face_feature(self):
        self.embeddings_data = pickle.loads(open("embeddings.pickle", "rb").read())
        self.diagnostic_logging(self.embeddings_data)
        self.diagnostic_logging("[INFO] encoding labels...")
        face_label_encoder = LabelEncoder()
        labels = face_label_encoder.fit_transform(self.embeddings_data["names"])

        self.diagnostic_logging("[INFO] training model...")
        face_feature = SVC(C=1.0, kernel="linear", probability=True)
        face_feature.fit(self.embeddings_data["embeddings"], labels)

        file = open("face_feature.pickle", "wb")
        file.write(pickle.dumps(face_feature))
        file.close()

        file = open("face_label_encoder.pickle", "wb")
        file.write(pickle.dumps(face_label_encoder))
        file.close()