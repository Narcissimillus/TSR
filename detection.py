import cv2 as cv
import numpy as np
from helpers import adjust_contrast_brightness, imadjust, union, intersection, inclusion

def main():
    # Other input parameters
    isNight = False

    # Your image path i-e receipt path
    img = cv.imread('GTSDB/00000.ppm')
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Enhance contrast
    rgb[:,:,0] = cv.medianBlur(rgb[:,:,0],3) #applying median filter to remove noice
    rgb[:,:,1] = cv.medianBlur(rgb[:,:,1],3) #applying median filter to remove noice
    rgb[:,:,2] = cv.medianBlur(rgb[:,:,2],3) #applying median filter to remove noice

    # Check road conditions
    if isNight:
        rgb = adjust_contrast_brightness(rgb, 2.2, 0)

    arr2 = rgb.copy()
    arr2 = cv.normalize(arr2.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    R_norm = arr2[:,:,0]
    G_norm = arr2[:,:,1]
    B_norm = arr2[:,:,2]

    # Normalize image for each channel
    R_norm = imadjust(R_norm, R_norm.min(), R_norm.max(), 0, 1) 
    G_norm = imadjust(G_norm, G_norm.min(), G_norm.max(), 0, 1)
    B_norm = imadjust(B_norm, B_norm.min(), B_norm.max(), 0, 1)

    # Normalize intensity of the red channel
    R_intensity = np.maximum(0,np.divide(np.minimum((R_norm - B_norm), (R_norm - G_norm)), (R_norm + G_norm + B_norm)))
    R_intensity[np.isnan(R_intensity)] = 0
    R_gray = (cv.normalize(R_intensity.astype('float'), None, 0, 255, cv.NORM_MINMAX)).astype('int')
    _, thresh_red = cv.threshold(R_gray.astype('uint8'), np.max(R_gray.astype('uint8')) - 80, 255, cv.THRESH_BINARY)
    # Normalize intensity of the blue channel
    B_intensity = np.maximum(0, np.divide((B_norm - R_norm),(R_norm + G_norm + B_norm)))
    B_intensity[np.isnan(B_intensity)] = 0
    B_gray = (cv.normalize(B_intensity.astype('float'), None, 0, 255, cv.NORM_MINMAX)).astype('int')
    _, thresh_blue = cv.threshold(B_gray.astype('uint8'), np.max(B_gray.astype('uint8')) - 80, 255, cv.THRESH_BINARY)

    contrast = adjust_contrast_brightness(img, 2.4, 105)

    # Convert BGR to HSV
    hsv = cv.cvtColor(contrast, cv.COLOR_BGR2HSV)

    # Equalize the histogram of the Y channel
    # hsv[:,:,0] = cv.equalizeHist(hsv[:,:,0])

    # Define range of blue color in HSV
    lower_blue = np.array([93, 163, 121])
    upper_blue = np.array([143, 255, 178])

    # Define range of red color in HSV
    # Note that red color appears 2 times in the hue channel so we use 2 masks
    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

    # Threshold the HSV image to get only red colors
    mask_red_1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv.inRange(hsv, lower_red_2, upper_red_2)

    # This is used to find only blue/red regions, remaining are ignored
    # For displaying purposes
    blue_only = cv.bitwise_and(hsv, hsv, mask = mask_blue)
    mask_red = cv.bitwise_or(mask_red_1, mask_red_2)
    red_only = cv.bitwise_and(hsv, hsv, mask = mask_red)

    kernel = np.ones((5,5), np.uint8)

    # Merge the red and blue masks
    mask_proposed = cv.bitwise_or(mask_blue, thresh_red)

    # Enchance features by dilating
    mask_proposed = cv.dilate(mask_proposed, kernel, iterations=1)

    # Display mask proposed by the algorithm
    cv.imshow('mask', mask_proposed)

    cv.waitKey(0)

    # Create MSER object
    mser = cv.MSER_create(min_area = 350, max_area = 10000)

    # Detect regions in gray scale image
    regions, boundingBoxes = mser.detectRegions(mask_proposed)

    # Compute hulls for every region
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    mask = np.zeros(mask_proposed.shape, np.uint8)
    boxes = []

    # Get only the bounding boxes that are not that overlapped
    for i, contour in enumerate(hulls):
        box = cv.boundingRect(contour)
        boxes.append(box)
        if i > 0:
            # If one box is included in another, merge them
            if inclusion(boxes[i], boxes[i - 1]) or inclusion(boxes[i - 1], boxes[i]):
                boxes[i] = union(boxes[i], boxes[i - 1])
                boxes[i - 1] = (0, 0, 0, 0)
            elif intersection(boxes[i], boxes[i - 1]) != ():
                _, _, w1, h1 = boxes[i]
                _, _, w2, h2 = boxes[i - 1]
                _, _, w, h = intersection(boxes[i], boxes[i - 1])
                # Merge only if the shapes overlaps are bigger than half of the biggest area
                if w * h >= w1 * h1 / 2 or w * h >= w2 * h2 / 2:
                    boxes[i] = union(boxes[i], boxes[i - 1])
                    boxes[i - 1] = (0, 0, 0, 0)

    vis = img.copy()
    # cv.polylines(vis, hulls, 1, (0, 255, 0))

    # This is used to find only sign regions, remaining are ignored
    for box in boxes:
            x, y, w, h = box
            if box != (0, 0, 0, 0):
                ar = w / float(h)
                # check if box is square-like
                if ar >= 0.75 and ar <= 1.25:
                    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display detection over the original image
    cv.imshow('img', vis)

    cv.waitKey(0)

    # Crop the detection and output it
    i = 0
    for box in boxes:
        x, y, w, h = box
        if box != (0, 0, 0, 0):
            ar = w / float(h)
            # Check if box is square-like
            if ar >= 0.75 and ar <= 1.25:
                i += 1
                cv.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                sign_only = cv.bitwise_and(img, img, mask=mask)
                cv.imwrite('{}.png'.format(i), img[y:y+h,x:x+w])
                cv.imshow("sign no " + str(i), img[y:y+h,x:x+w])

            cv.waitKey(0)

if __name__ == "__main__":
    main()
