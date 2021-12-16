import cv2 as cv
import sys
import numpy as np
from helpers import adjust_contrast_brightness

def do_nothing(x):
    pass

# Load in image
image = cv.imread('GTSDB/00006.jpg')

# Create a window
cv.namedWindow('image', cv.WINDOW_NORMAL)

# Create trackbars for color change
cv.createTrackbar('HMin', 'image', 0, 179, do_nothing) # Hue is from 0-179 for Opencv
cv.createTrackbar('SMin', 'image', 0, 255, do_nothing)
cv.createTrackbar('VMin', 'image', 0, 255, do_nothing)
cv.createTrackbar('HMax', 'image', 0, 179, do_nothing)
cv.createTrackbar('SMax', 'image', 0, 255, do_nothing)
cv.createTrackbar('VMax', 'image', 0, 255, do_nothing)
cv.createTrackbar('Contrast', 'image', 100, 400, do_nothing)
cv.createTrackbar('Brightness', 'image', 256, 511, do_nothing)

# Set default value for MAX HSV trackbars
cv.setTrackbarPos('HMax', 'image', 179)
cv.setTrackbarPos('SMax', 'image', 255)
cv.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0
contrast = 0
brightness = 0

output = image
wait_time = 33

while(1):

    # Get current positions of all trackbars
    hMin = cv.getTrackbarPos('HMin','image')
    sMin = cv.getTrackbarPos('SMin','image')
    vMin = cv.getTrackbarPos('VMin','image')

    hMax = cv.getTrackbarPos('HMax','image')
    sMax = cv.getTrackbarPos('SMax','image')
    vMax = cv.getTrackbarPos('VMax','image')

    contrast = cv.getTrackbarPos('Contrast', 'image')
    brightness = cv.getTrackbarPos('Brightness', 'image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range
    # Enhance contrast
    cont = adjust_contrast_brightness(image, contrast / 100, brightness - 255)
    hsv = cv.cvtColor(cont, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    output = cv.bitwise_and(image, image, mask= mask)

    # Display output image
    cv.imshow('image', output)

    # Wait longer to prevent freeze for videos
    if cv.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()