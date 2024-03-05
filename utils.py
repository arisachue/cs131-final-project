import cv2
import numpy as np


def detect_face_coords(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)

    biggest = (0, 0, 0, 0)
    for i in coords:
        if i[3] > biggest[3]:
            biggest = i
    return biggest


def detect_eyes_coords(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = (0, 0, 0, 0)
    right_eye = (0, 0, 0, 0)
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = (x, y, w, h)
        else:
            right_eye = (x, y, w, h)
    return left_eye, right_eye

def remove_eyebrows(img):
    height, width = img.shape[: 2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h : height, 0 : width]
    
    return img

def find_eye_keypoints(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
    
#     img = cv2.erode(img, None, iterations=2) #1
#     img = cv2.dilate(img, None, iterations=4) #2
#     img = cv2.medianBlur(img, 5) #3

    keypoints = detector.detect(img)
    return keypoints