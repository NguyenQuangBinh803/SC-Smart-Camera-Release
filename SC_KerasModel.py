import face_feature_v4
import threading
import cv2
from multiprocessing import Process
import time
import tensorflow as tf



# print(input_shape)


#
#
# # maskNet = load_model()
# converter = tf.lite.TFLiteConverter.from_keras_model_file("mask_detector.model")
# # converter = tf.lite.TFLiteConverter.from_saved_model("mask_detector.model") # path to the SavedModel directory
# maskNet = converter.convert()
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(maskNet)
# # maskNet._make_predict_function()

if __name__ == "__main__":
    # p = Process(target=grab_image).start()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            face_feature_v4.detect_faces(frame, frame)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

