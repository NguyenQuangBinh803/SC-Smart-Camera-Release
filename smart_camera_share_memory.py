from tensorflow import keras
import tensorflow as tf
import cv2
import time
import datetime

print("Init parameter ... ")
# load our serialized face detector model from disk
# prototxtPath = "face_detector/deploy.prototxt"
# weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
#
# faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
# config = tf.ConfigProto(
#     device_count={'GPU': 1},
#     intra_op_parallelism_threads=1,
#     allow_soft_placement=True
# )
# session = tf.Session(config=config)
#
# keras.backend.set_session(session)
#
# maskNet = tf.lite.Interpreter(model_path="SmartCameraModel/model.tflite")
# maskNet.allocate_tensors()
#
# input_details = maskNet.get_input_details()
# output_details = maskNet.get_output_details()
# input_shape = input_details[0]['shape']

time.sleep(3)
thermal_data = 36.5


global_locs = []
global_mask = []

face_detect_status = False
mask_detect_status = False
human_appear_status = False

frame = None
thermal = None
frame_face = {}
frame_face["frame"] = frame
frame_face["thermal"] = thermal

global_face_image = None
global_human_name = None

face_area = 0

print("Done init parameters ... ")

