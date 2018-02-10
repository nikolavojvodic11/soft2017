import cv2
import numpy as np


def houghTransformtion(frame, grayImg):
    edges = cv2.Canny(grayImg, 50, 150, apertureSize=3)  # canny vraca ivice cele slike
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, 600, 8)  # vraca vektora linija i smest ih u linije

    minx = 1500
    miny = 1500
    maxy = -5
    maxx = -5

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if x2 > maxx:  # and y2>52
                maxy = y2
                maxx = x2
            if x1 < minx:
                minx = x1
                miny = y1

    return minx, miny, maxx, maxy
