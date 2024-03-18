import cv2
import numpy as np


def detect_face_coords(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # TODO add parameter optimization
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)

    biggest = (0, 0, 0, 0)
    for i in coords:
        if i[3] > biggest[3]:
            biggest = i
    return biggest


def detect_eyes_coords(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = (0, 0, 0, 0)
    right_eye = (0, 0, 0, 0)
    
    # find the best detectMultiScale parameters
    scaleFactors = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5]
    minNeighbors = [3, 4, 5, 6]
    for sf in scaleFactors:
        for mn in minNeighbors:
            eyes = classifier.detectMultiScale(gray_frame, sf, mn, 0, (int(width/5), int(height/5))) # detect eyes
            if len(eyes) >= 2:
                if abs(eyes[0][1] - eyes [1][1]) < 50:
                    for (x, y, w, h) in eyes:
                        if y < height / 2: 
                            eyecenter = x + w / 2  # get the eye center
                            if eyecenter < width * 0.5:
                                left_eye = (x, y, w, h)
                            else:
                                right_eye = (x, y, w, h)
                    return left_eye, right_eye
    
    '''eyes = classifier.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    for (x, y, w, h) in eyes:
        
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = (x, y, w, h)
        else:
            right_eye = (x, y, w, h)'''
    return (0, 0, 0, 0), (0, 0, 0, 0)

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