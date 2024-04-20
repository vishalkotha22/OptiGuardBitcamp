import cv2
import threading
import torch
from collections import deque
import numpy as np
import tensorflow as tf
from face_detector import YoloDetector
import numpy as np

CLASSES_LIST = ["NonViolence", "Violence"]

def camPreview(previewName, camID):
    #cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)

    if camID == 1:
        model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Resize the frame (optional)
            frame = cv2.resize(frame, (640, 480))
            orgimg = np.array(frame)
            bboxes, points = model.predict(orgimg)
            #print(points)
            #print(bboxes)

            if ([] not in bboxes):
                for tempArr in bboxes[0]:
                    print(tempArr)
                    cv2.rectangle(frame, (tempArr[0], tempArr[1]), (tempArr[2], tempArr[3]), (0, 255, 0), 2)

            cv2.imshow("Camera " + str(camID), frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        model = tf.keras.models.load_model('model.h5')

        frames_queue = deque(maxlen=16)
        predicted_class_name = ''

        while True:
            ret, frame = cam.read()

            if not ret:
                break

            # Resize the frame (optional)
            tempframe = cv2.resize(frame, (64, 64))
            # h, w, c = frame.shape
            # print(h, w, c)
            normalized_frame = tempframe / 255
            # features = np.asarray(normalized_frame)
            frames_queue.append(normalized_frame)
            if len(frames_queue) == 16:
                # currPrediction = model.predict(features)
                predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                if predicted_labels_probabilities[1] > 0.05:
                    predicted_class_name = "Violence"
                else:
                    predicted_class_name = "NonViolence"
                #predicted_label = np.argmax(predicted_labels_probabilities)

                # Get the class name using the retrieved index.
                #predicted_class_name = CLASSES_LIST[predicted_label]

            if predicted_class_name == "Violence":
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
            else:
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

            cv2.imshow("Camera " + str(camID), frame)
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

# Create threads for each camera
thread1 = threading.Thread(target=camPreview, args=("Camera 1", 0))
thread2 = threading.Thread(target=camPreview, args=("Camera 2", 1))

# Start the threads
thread1.start()
thread2.start()

# Wait for threads to finish
thread1.join()
thread2.join()