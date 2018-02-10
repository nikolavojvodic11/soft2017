import numpy as np
import cv2


broj = 0
prosli = []

def Line(video_Name):

    import HoughTransformations as ht
    cap = cv2.VideoCapture(video_Name)
    kernel = np.ones((2, 2), np.uint8)

    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img0 = cv2.dilate(gray, kernel)

        cap.release()
        cv2.destroyAllWindows()
    return ht.houghTransformtion(frame,img0)
