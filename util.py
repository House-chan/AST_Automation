import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import copy


def show(img, cmap="gray", title="", xlabel="", ylabel=""):
    plt.imshow(img, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    

def draw_circle(img, circles, color=(0, 255, 0), thickness=3, pad=0):
    """
    draw a circles on a images(img)
    """
    img = copy(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # Draw circles that are detected. 
    if circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        circles = np.uint16(np.around(circles)) 
    
        for pt in circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            r = r + pad
            # Draw the circumference of the circle. 
            img = cv2.circle(img, (a, b), r, color, thickness) 
    
            # Draw a small circle (of radius 1) to show the center. 
            img = cv2.circle(img, (a, b), 1, color, thickness) 
            

    return img


def resize(img, target_width, fit_width=True):
    """resize an image with the same ratio
    if fit_width is true, width of the resized image equal to target_width
    """
    w, h = img.shape

    m_s = min(w,h) if fit_width  else max(w,h)

    width = int(img.shape[1] * target_width / m_s)
    height = int(img.shape[0] * target_width / m_s)

    dim = (width, height)
    img = cv2.resize(img, dim)
    return img  


def boundary(val, min ,max):
    if val < min:
        return min
    elif val > max:
        return max
    else:
        return val


def get_pellet_list():
    PELLET_LIST = [
    'AK 30', 'AM 2', 'AMC 30', 'AMP 10', 'AMP 2', 'ATM 30', 'AUG 30',
    'AX 20', 'C 30', 'CAZ 10', 'CD 2', 'CFM 5', 'CFR 30', 'CIP 5',
    'CN 10', 'CN 30', 'CN 500', 'CRO 30', 'CT 50', 'CTX 5', 'E 15',
    'ERY 15', 'ETP 10', 'FC 10', 'F 100', 'FEC 40', 'FEP 30', 'FF 200',
    'FOX 30', 'IMI 10', 'IPM 10', 'L 15', 'LEV 5', 'LNZ 10', 'LVX 5',
    'MEC 10', 'MEM 10', 'MRP 10', 'MXF 5', 'NA 30', 'NET 10', 'NOR 10',
    'OX 1', 'P 1', 'PRL 30', 'PT 15', 'RA 5', 'RD 5', 'S 300',
    'SXT 25', 'TC 75', 'TEC 30', 'TEM 30', 'TET 30', 'TGC 15', 'TIC 75',
    'TIM 85', 'TOB 10', 'TPZ 36', 'TTC 85', 'TZP 36', 'VA 30', 'VA 5']
    REMOVED_CLASSES = ['AK 30', 'RD 5', 'FEC 40', 'FC 10', 'TGC 15']
    return PELLET_LIST

def adjust_brightness(image, target_brightness):
    # Compute average brightness
    avg_brightness = np.mean(image)

    # Compute adjustment factor
    adjustment_factor = target_brightness / avg_brightness

    # Adjust brightness of image
    adjusted_image = np.clip(image * adjustment_factor, 0, 255).astype(np.uint8)

    return adjusted_image

def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
    return moving_averages
