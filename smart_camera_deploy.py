#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_camera_ui import *

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    camera_capture_thread = threading.Thread(target=grab_image, args=[-1, 2, ]).start()
    detect_face_and_mask_capture_thread = threading.Thread(target=detect_face_and_mask).start()
    face_capture_thread = threading.Thread(target=grab_face_in).start()

    main_window = Window()
    sys.exit(app.exec_())