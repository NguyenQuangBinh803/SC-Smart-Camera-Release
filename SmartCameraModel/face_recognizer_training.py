__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"
__email__ = "nguyenquangbinh803@gmail.com"
__copyright__ = "Copyright 2020"
__status__ = "Working on embedding recognition to smart camera"
__version__ = "1.0.1"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob

# import face_recognition

from smart_camera_common_imports import *
from SmartCameraModel.face_detect_and_recognize import FaceDetectAndRecognition


class FaceRecognizerTraining:

    def __init__(self):
        self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    def encoding_with_torch_openface(self, dataset_directory):
        folders = glob.glob(os.path.abspath("") + dataset_directory + "/*/")
        knownNames = []
        knownEmbeddings = []

        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))
            # print(target_name)
            images = glob.glob(folder + "\\*.jpg")
            # images = glob.glob("SmartCameraModel/dataset/" + target_name + "/" + "*.jpg")
            print(folder, len(images))
            total = 0
            for image in images:
                img = cv2.imread(image)
                start = time.time()
                faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()
                print(target_name, time.time() - start)
                knownNames.append(target_name)
                knownEmbeddings.append(vec.flatten())
                total += 1

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open("embeddings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

    def encoding_with_dlib_face(self, dataset_directory):
        folders = glob.glob(os.path.abspath("") + "/" + dataset_directory + "/*/")
        knownNames = []
        knownEncodings = []
        print(folders)
        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))

            images = glob.glob(folder + "/*.jpg")
            print(folder, target_name, len(images))
            total = 0
            for image in images:
                img = cv2.imread(image)
                start = time.time()
                height, width = img.shape[:2]
                print(0, width, height, 0)
                boxes = [(0, width, height, 0)]
                encodings = face_recognition.face_encodings(img, boxes)
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(target_name)
                print(time.time() - start)
                total += 1

        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

    def training_svm_model_openface(self):
        self.embeddings_data = pickle.loads(open("embeddings.pickle", "rb").read())
        print(self.embeddings_data)
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(self.embeddings_data["names"])

        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(self.embeddings_data["embeddings"], labels)

        f = open("recognizer.pickle", "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        f = open("le.pickle", "wb")
        f.write(pickle.dumps(le))
        f.close()

    def recognize_with_openface(self, face):
        recognizer = pickle.loads(open("recognizer.pickle", "rb").read())
        le = pickle.loads(open("le.pickle", "rb").read())

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]


if __name__ == "__main__":
    face_encode = FaceRecognizerTraining()
    # face_recognize = FaceDetectAndRecognition()
    face_encode.encoding_with_torch_openface("\\dataset\\")
    face_encode.training_svm_model_openface()
