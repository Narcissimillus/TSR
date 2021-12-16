import cv2 as cv

def imadjust(x, a, b, c, d, gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def adjust_contrast_brightness(img, contrast:float = 1.0, brightness:int = 0):
    # Adjusts contrast and brightness with OpenCV addWeighted built-in function
    # contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    # brightness: [-255, 255] with 0 leaving the brightness as is
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv.addWeighted(img, contrast, img, 0, brightness)

def intersection(a,b):
    # Check intersection of the a and b shapes
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def inclusion(a, b): # is a included in b?
    # Return True if and only if whole shape a is included in shape b
    if a[0] >= b[0] and (a[0] + a[2]) <= (b[0] + b[2]) and  \
       a[1] >= b[1] and (a[1] + a[3]) <= (b[1] + b[3]):
        return True
    return False

def union(a, b):
    # Return a merged shape of the input
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)