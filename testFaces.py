#import sys
#sys.path.append(".\\yoloface-master\\face_detector.py")
#import os
#os.chdir("/yoloface-master")

from face_detector import YoloDetector
import numpy as np
import cv2

cv2.namedWindow("Camera One")
cam = cv2.VideoCapture(0)
model = YoloDetector(target_size=720, device="cpu", min_face=90)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Resize the frame (optional)
    frame = cv2.resize(frame, (640, 480))
    orgimg = np.array(frame)
    bboxes,points = model.predict(orgimg)
    print(points)
    print(bboxes)

    if([] not in bboxes):
        for tempArr in bboxes[0]:
            print(tempArr)
            cv2.rectangle(frame, (tempArr[0], tempArr[1]), (tempArr[2], tempArr[3]), (0, 255, 0), 2)

    cv2.imshow("Camera One", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


