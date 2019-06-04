import cv2 as cv
import numpy as np

# Capture the input frame from webcam


cap = cv.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 15
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([180, sensitivity, 255])

    # Threshold the HSV image to get only white colors
    mask = cv.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('res', res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
