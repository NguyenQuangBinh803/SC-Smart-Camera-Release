from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("smart_camera_share_memory", ["smart_camera_share_memory.py"]),
    Extension("smart_camera_process", ["smart_camera_process.py"]),
    Extension("smart_camera_ui", ["smart_camera_ui.py"]),
]
setup(
    name='My Program Name',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
