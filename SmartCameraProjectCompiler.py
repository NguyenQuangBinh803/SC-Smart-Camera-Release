from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("smart_camera_share_memory", ["SmartCameraShareMemory.py"]),
    Extension("smart_camera_process", ["SmartCameraProcess.py"]),
    Extension("smart_camera_ui", ["SmartCameraUi.py"]),
]
setup(
    name='My Program Name',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
