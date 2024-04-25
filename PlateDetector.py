import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from util import resize


def resize(img, target_width, fit_width=True):
    """resize an image with the same ratio
    if fit_width is true, width of the resized image equal to target_width
    """
    w, h = img.shape

    m_s = min(w,h) if fit_width  else max(w,h)
    if m_s == 0:
        return img

    width = int(img.shape[1] * target_width / m_s)
    height = int(img.shape[0] * target_width / m_s)

    dim = (width, height)
    img = cv2.resize(img, dim)
    return img  


def detect(image):
    # determine the max and min size of the plate
    maxRadius = int(0.9*max(image.shape)/2)
    minRadius = int(0.1*max(image.shape)/2)

    # decrease the detail of the image.
    kernel2 = np.ones((5, 5), np.uint8)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    iter = 4
    dilation = cv2.dilate(blurred_image, kernel2, iterations=iter)
    erosion = cv2.erode(dilation, kernel2, iterations=iter)
    
    # Use the Circular Hough Transform to detect the biggest circle
    circles = cv2.HoughCircles(
        erosion, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=120, param2=100, minRadius=minRadius, maxRadius=maxRadius
    )
    try:
        # Your processing code here
        
        if circles is not None:
            # Sort by radius and get the largest circle
            indx = np.argsort(circles[:, :, -1])
            biggest_circles = circles[:, indx[:, -1]]

            # Make circle a bit smaller
            # biggest_circles[0,0,2] = biggest_circles[0,0,2] - 40
            return biggest_circles
        else:
           
            return detect_sec(image)

    except cv2.error as e:
        print("An OpenCV error occurred:", e)
        return "An OpenCV error occurred:", e  # Or handle the error differently

    except Exception as e:
        print("No Plate detected:", e)
        return "No Plate detected:", e


def detect_sec(image):
    # determine the max and min size of the plate
    maxRadius = int(0.9*max(image.shape)/2)
    minRadius = int(0.1*max(image.shape)/2)

    # decrease the detail of the image.
    kernel2 = np.ones((3, 2), np.uint8)
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    iter = 2
    dilation = cv2.dilate(blurred_image, kernel2, iterations=iter)
    erosion = cv2.erode(dilation, kernel2, iterations=iter)

    # Calculate gradients in x and y directions
    grad_x = cv2.Sobel(erosion, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(erosion, cv2.CV_64F, 0, 1, ksize=3)  

    # Convert gradients back to uint8 for combining and displaying
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Combine gradients 
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Use the Circular Hough Transform to detect the biggest circle
    circles = cv2.HoughCircles(
        grad, cv2.HOUGH_GRADIENT, dp=1, minDist=49, param1=80, param2=200, minRadius=minRadius, maxRadius=maxRadius
    )
    try:
        if circles is not None:
            # Sort by radius and get the largest circle
            indx = np.argsort(circles[:, :, -1])
            biggest_circles = circles[:, indx[:, -1]]

            # Make circle a bit smaller
            # biggest_circles[0,0,2] = biggest_circles[0,0,2] - 40
            return biggest_circles
        else:
            return  detect_third(image)

    
    except cv2.error as e:
        print("An OpenCV error occurred:", e)
        return "An OpenCV error occurred:", e  # Or handle the error differently

    except Exception as e:
        print("No Plate detected:", e)
        return "No Plate detected:", e



def detect_third(image):
    # determine the max and min size of the plate
    maxRadius = int(0.9*max(image.shape)/2)
    minRadius = int(0.1*max(image.shape)/2)

    # decrease the detail of the image.
    kernel2 = np.ones((2, 2), np.uint8)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    iter = 2
    dilation = cv2.dilate(blurred_image, kernel2, iterations=iter)
    erosion = cv2.erode(dilation, kernel2, iterations=iter)
    
    # Use the Circular Hough Transform to detect the biggest circle
    circles = cv2.HoughCircles(
        erosion, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=30, param2=60, minRadius=minRadius, maxRadius=maxRadius
    )
    try:
        if circles is not None:
            # Sort by radius and get the largest circle
            indx = np.argsort(circles[:, :, -1])
            biggest_circles = circles[:, indx[:, -1]]

            # Make circle a bit smaller
            # biggest_circles[0,0,2] = biggest_circles[0,0,2] - 40
            return biggest_circles
        else: None
    
    except cv2.error as e:
        print("An OpenCV error occurred:", e)
        return "An OpenCV error occurred:", e  # Or handle the error differently

    except Exception as e:
        print("No Plate detected:", e)
        return "No Plate detected:", e
    

def circle_crop(image, circle, pad=0, normalize_size=True):
    
    def boundary(val, min ,max):
        if val < min:
            return min
        elif val > max:
            return max
        else:
            return val
    
    x, y, r = np.round(circle).astype(int)[0,0]
    
    w, h = image.shape

    ym = boundary(y-r-pad, 0, w)
    yM = boundary(y+r+pad, 0, w)
    xm = boundary(x-r-pad, 0, h)
    xM = boundary(x+r+pad, 0, h)

    img_crop = image[ym: yM, xm: xM]
    
    # normalize image size to equla input image size
    size = min(w,h)
    img_crop = cv2.resize(img_crop, (size,size)) if normalize_size else img_crop

    return img_crop


def detect_crop(image, pad=3, normalize_size=True):
    image = resize(image, 500)

    # find circle of the plate.
    circle = detect(image)

    img_crop = circle_crop(image, circle, pad, normalize_size)
    
    return(img_crop)