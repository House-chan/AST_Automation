import cv2
import numpy as np
from os import path
from copy import copy
import PlateDetector
from util import show, draw_circle, resize

def detect(img, pad=0):
    # crop image at place
    
    img_blur = cv2.GaussianBlur(img, (5, 5),0)
    normalized_img = cv2.normalize(img_blur, None, 20, 350, cv2.NORM_MINMAX)
    iter = 3
    kernel2 = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img_blur, kernel2, iterations=iter)
    erosion = cv2.erode(dilation, kernel2, iterations=iter)

    # _, img_th = cv2.threshold(erosion,190,255,cv2.THRESH_BINARY)


    detected_circles = cv2.HoughCircles(erosion,  
                    cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1 = 60, 
                param2 = 20, minRadius = 62, maxRadius = 74) 
    
    # expand the circles
    detected_circles[:,:,2] = detected_circles[:,:,2] + pad

    return detected_circles