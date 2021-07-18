__author__ = 'Edward J. C. Ashenbert'
__credits__ = ["Edward J. C. Ashenbert"]
__maintainer__ = "Edward J. C. Ashenbert"
__email__ = "nguyenquangbinh803@gmail.com"
__copyright__ = "Copyright 2020"
__status__ = "Working on embedding recognition to smart camera"
__version__ = "1.4 - Fix a lot of unbelievably terrible design - 20210718"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import face_recognition
import SmartCameraShareMemory as sc_share_memory
from SmartCameraCommonImports import *

class FaceDetectAndRecognition:

    def __init__(self):

        self.diagnostic_logging("Init models")
        
        self.face_features = pickle.loads(open(Path_FaceRecognizerFeatureList, "rb").read())
        self.face_labels = pickle.loads(open(Path_FaceRecognizerLabelList, "rb").read())

        self.face_features_extractor = cv2.dnn.readNetFromTorch(Path_FaceRecognizerFeatureExtractor)

        # Initialize resnet face_image detector
        self.face_detector = cv2.dnn.readNetFromCaffe(Path_FaceDetectorProto, Path_FaceDetectorWeights)

        # Initialize keras mask detector
        self.session = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
        ))
        
        keras.backend.set_session(self.session)

        # Explanation of this arrangement is that the flow of initialization and process flow is 
        # quite complex 

        self.face_mask_detector = tf.lite.Interpreter(model_path=Path_MaskDetectorModel)
        self.face_mask_detector.allocate_tensors()

        self.input_information = self.face_mask_detector.get_input_details()
        self.input_shape = self.input_information[0]['shape']
        self.output_information = self.face_mask_detector.get_output_details()

        self.diagnostic_logging("Done init models")


    def diagnostic_logging(self, message):
        print("[" + datetime.datetime.strftime("%Y%m%d-%H%M%S", datetime.datetime.now()) + "] " + message)


    def executing_face_detect(self, input_image, rgb_require=True):

        input_image_height, input_image_width = input_image.shape[:2]
        input_image_blob = cv2.dnn.blobFromImage(input_image, 1.0, (224, 224), (104.0, 177.0, 123.0))

        self.face_detector.setInput(input_image_blob)
        face_detect_results = self.face_detector.forward()
        face_detect_list = []
        face_detect_location_list = []
        face_image_rgb = []
        face_image_normal = []

        for i in range(0, face_detect_results.shape[2]):
            detect_confidence = face_detect_results[0, 0, i, 2]
            if detect_confidence > 0.5:
                sc_share_memory.human_appear_status = True
                box = face_detect_results[0, 0, i, 3:7] * np.array([input_image_width, input_image_height, input_image_width, input_image_height])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(input_image_width - 1, endX), min(input_image_height - 1, endY))
                try:
                    face_image = input_image[startY:endY, startX:endX]
                    face_image_normal.append(face_image)
                    cv2.imwrite(Path_FaceTemporaryImage + str(datetime.now().strftime("%Y%m%d")) + ".jpg", face_image)
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    face_image = cv2.resize(face_image, (224, 224))
                    face_image_rgb.append(face_image)
                    face_image = img_to_array(face_image)
                    face_image_processed = preprocess_input(face_image)
                    

                    face_detect_list.append(face_image_processed)
                    sc_share_memory.global_face_image = face_detect_list
                    sc_share_memory.face_area = (endX - startX) * (endY - startY)
                    if (endX - startX) * (endY - startY) > 3000:
                        # This should be implement as a ShareMemory Class
                        sc_share_memory.face_detect_status = True
                    else:
                        sc_share_memory.face_detect_status = False

                    face_detect_location_list.append((startX, startY, endX, endY))
                except Exception as exp:
                    self.diagnostic_logging(str(exp))


        if not face_detect_location_list:
            sc_share_memory.face_detect_status = False
            sc_share_memory.human_appear_status = False

        if sc_share_memory.face_detect_status:
            sc_share_memory.global_locs = face_detect_location_list

        if rgb_require:
            return face_detect_location_list, face_image_rgb
        else:
            return face_detect_location_list, face_image_processed


    def executing_mask_detect(self, input_face_image):
        try:
            with self.session.as_default():
                with self.session.graph.as_default():
                    face_detect_list = np.array(input_face_image, dtype="float32")
                    self.face_mask_detector.set_tensor(self.input_information[0]['index'], face_detect_list)
                    self.face_mask_detector.invoke()
                    mask_detect_list = self.face_mask_detector.get_tensor(self.output_information[0]['index'])
                    self.diagnostic_logging(mask_detect_list)
                        
                    if len(mask_detect_list[0]) > 1 and mask_detect_list[0][1] > 0.5:
                        sc_share_memory.mask_detect_status = False
                        return False
                    elif len(mask_detect_list[0]) > 1 and mask_detect_list[0][0] > 0.5:
                        sc_share_memory.mask_detect_status = True
                        return True
                    
        except Exception as exp:
            self.diagnostic_logging(exp)


    def executing_face_and_mask_detect(self, input_image):
        # Should bind data to the share memory here
        output_face_image = self.executing_face_detect(input_image, rgb_require=False)[1]
        output_mask_result = self.executing_mask_detect(output_face_image)


    def executing_face_detect_and_mask(self, input_image):
        input_image_height, input_image_width = input_image.shape[:2]
        input_image_blob = cv2.dnn.blobFromImage(input_image, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.face_detector.setInput(input_image_blob)
        face_detect_results = self.face_detector.forward()

        face_detect_list = []
        face_detect_location_list = []
        mask_detect_list = []

        for i in range(0, face_detect_results.shape[2]):
            detect_confidence = face_detect_results[0, 0, i, 2]
            if detect_confidence > 0.5:
                sc_share_memory.human_appear_status = True
                box = face_detect_results[0, 0, i, 3:7] * np.array([input_image_width, input_image_height, input_image_width, input_image_height])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(input_image_width - 1, endX), min(input_image_height - 1, endY))

                face_image = input_image[startY:endY, startX:endX]
                sc_share_memory.global_face_image = face_image
                cv2.imwrite( + str(datetime.now().strftime("%Y%m%d")) + ".jpg", face_image)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image = cv2.resize(face_image, (224, 224))
                face_image = img_to_array(face_image)
                face_image = preprocess_input(face_image)

                face_detect_list.append(face_image)
                
                sc_share_memory.face_area = (endX - startX) * (endY - startY)

                if (endX - startX) * (endY - startY) > 3000:
                    sc_share_memory.face_detect_status = True
                else:
                    sc_share_memory.face_detect_status = False

                face_detect_location_list.append((startX, startY, endX, endY))

        if not face_detect_location_list:
            sc_share_memory.face_detect_status = False
            sc_share_memory.human_appear_status = False

        if sc_share_memory.face_detect_status:
            sc_share_memory.global_locs = face_detect_location_list

        if len(face_detect_list) > 0:
            try:
                with self.session.as_default():
                    with self.session.graph.as_default():
                        face_detect_list = np.array(face_detect_list, dtype="float32")
                        self.face_mask_detector.set_tensor(self.input_information[0]['index'], face_detect_list)
                        self.face_mask_detector.invoke()
                        mask_detect_list = self.face_mask_detector.get_tensor(self.output_information[0]['index'])
                        self.diagnostic_logging(mask_detect_list)
                            
                        if len(mask_detect_list[0]) > 1 and mask_detect_list[0][1] > 0.5:
                            sc_share_memory.mask_detect_status = False
                        elif len(mask_detect_list[0]) > 1 and mask_detect_list[0][0] > 0.5:
                            sc_share_memory.mask_detect_status = True
                        
            except Exception as exp:
                self.diagnostic_logging(exp)

        return (face_detect_location_list, mask_detect_list)


    def executing_face_recognize(self, face_image):
        faceBlob = cv2.dnn.blobFromImage(face_image, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.face_features_extractor.setInput(faceBlob)
        face_feature_extraction_results = self.face_features_extractor.forward()
        mask_detect_list = self.face_features.predict_proba(face_feature_extraction_results)[0]
        proba = mask_detect_list[np.argmax(mask_detect_list)]
        name = self.face_labels.classes_[np.argmax(mask_detect_list)]

        self.diagnostic_logging([name, proba])

        if proba > 0.5:
            return name
        else:
            return "Unknown"


    def training_encode_face_image(self, dataset_directory):
        folders = glob.glob(os.path.abspath("") + dataset_directory + "/*/")
        knownNames = []
        knownEmbeddings = []

        for folder in folders:
            target_name = os.path.basename(os.path.normpath(folder))
            images = glob.glob(folder + "\\*.jpg")
            self.diagnostic_logging(folder, len(images))
            total = 0
            for image in images:
                img = cv2.imread(image)
                start = time.time()
                faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.face_features_extractor.setInput(faceBlob)
                face_feature_extraction_results = self.face_features_extractor.forward()
                self.diagnostic_logging(target_name, time.time() - start)
                knownNames.append(target_name)
                knownEmbeddings.append(face_feature_extraction_results.flatten())
                total += 1

        data = {"embeddings": knownEmbeddings, "names": knownNames}
        file = open("embeddings.pickle", "wb")
        file.write(pickle.dumps(data))
        file.close()
        

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
    

    def testing_face_recognize(self, face_image):
        face_feature = pickle.loads(open("face_feature.pickle", "rb").read())
        face_label = pickle.loads(open("face_label.pickle", "rb").read())

        faceBlob = cv2.dnn.blobFromImage(face_image, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.face_features_extractor.setInput(faceBlob)
        face_feature_extraction_results = self.face_features_extractor.forward()
        mask_detect_list = face_feature.predict_proba(face_feature_extraction_results)[0]
        
        return mask_detect_list[np.argmax(mask_detect_list)], face_label.classes_[np.argmax(mask_detect_list)]


if __name__ == "__main__":
    face_detect = FaceDetectAndRecognition()