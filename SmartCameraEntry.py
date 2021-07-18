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
    threading.Thread(target=smart_camera_controller.threading_face_recognize).start()
    threading.Thread(target=smart_camera_controller.threading_detect_face).start()
    threading.Thread(target=smart_camera_controller.threading_calculate_temperature).start()

    main_window = Window()
    sys.exit(app.exec_())
