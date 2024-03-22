import cv2
import numpy as np


def detect_face_coords(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)

    biggest = np.zeros(4)
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
                if abs(eyes[0][1] - eyes[1][1]) < 50:
                    for (x, y, w, h) in eyes:
                        if y < height / 3: 
                            eyecenter = x + w / 2  # get the eye center
                            if eyecenter < width * 0.5:
                                left_eye = (x, y, w, h)
                            else:
                                right_eye = (x, y, w, h)
                    return left_eye, right_eye
    
    return (0, 0, 0, 0), (0, 0, 0, 0)

def remove_eyebrows(img):
    height, width = img.shape[: 2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h : height, 0 : width]
    
    return img

def find_eye_keypoints(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(img)
    return keypoints

def find_best_threshold(gray_img):
    start = 60
    
    _, eye_img = cv2.threshold(gray_img, start, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(eye_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # keeps decreasing threshold (make it stricter) until 2 contours remain (one is the whole window frame, one is the real contour)
    while len(contours) > 2 and start >= 5:
        start -= 2
        _, eye_img = cv2.threshold(gray_img, start, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(eye_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    return start + 2

# mainly shifting contours that get affected by makeup
def shift_contour_inside_eye(eye_img, contour):
    width = np.size(eye_img, 1) 
    height = np.size(eye_img, 0) 
    
    contour_coords = contour.reshape(-1, 2)

    # center contour point
    center_x = np.mean(contour_coords[:, 0])
    center_y = np.mean(contour_coords[:, 1])
    
    # getting approx boundaries of the eye
    start_approx_eye_x = width * 0.25
    start_approx_eye_y = height * 0.25
    end_approx_eye_x = width * 0.75
    end_approx_eye_y = width * 0.75
    center_eye_x = width / 2
    center_eye_y = height / 2
    
    shift_x = 0    
    if center_x < start_approx_eye_x or center_x > end_approx_eye_x:
        shift_x = center_eye_x - center_x
    shift_y = 0    
    if center_y < start_approx_eye_y or center_y > end_approx_eye_y:
        shift_y = center_eye_y - center_y
        
    # shifting the contour points towards center of eye
    contour = contour + [shift_x, shift_y]
    
    return contour

# for manual iris drawing, or drawing the contour directly
def is_contour_circular(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    circularity_ratio = 4 * np.pi * area / (perimeter ** 2)
    circularity_threshold = 0.6  # Adjust as needed
    
    return circularity_ratio >= circularity_threshold