import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import sqrt


def lines_from_img(orig_img, save_results=False):
    img = orig_img

    # gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # binarize
    binarized = gray
    # _, binarized = cv.threshold(binarized,190,255,cv.THRESH_BINARY)
    binarized = cv.GaussianBlur(binarized,(5,5),0)
    _,binarized = cv.threshold(binarized,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # binarized = cv.adaptiveThreshold(binarized,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    # binarized = cv.adaptiveThreshold(binarized,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    # remove noise
    noNoise = gray
    # opening / closing
    kernel = np.ones((5,5),np.uint8)
    noNoise = cv.morphologyEx(noNoise, cv.MORPH_OPEN, kernel)
    noNoise= cv.morphologyEx(noNoise, cv.MORPH_CLOSE, kernel)
    # blur
    noNoise= cv.GaussianBlur(noNoise, (5,5), 0)
    # noNoise = cv.blur(noNoise, (5,5))

    # thicc edges
    edges = noNoise
    edges = cv.Canny(edges, 100, 255)
    kernel = np.ones((2,2), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)

    lines = cv.HoughLinesP(edges, cv.HOUGH_PROBABILISTIC, np.pi/180, 10, minLineLength=10, maxLineGap=25)

    if save_results:
        save_image(orig_img, 'orig')
        save_image(edges, 'edges')
        save_image(binarized, 'bin')

    if type(lines) != np.ndarray:
        return []

    # flatten array by 1 dim
    amount, z, features = lines.shape
    lines = lines.reshape(amount*z, features)
    return lines


def draw_line(image, line, color=(0, 255, 0)):
    x1, y1, x2, y2 = line
    # cv.line(image, (x1, y1), (x2, y2), color, 10)
    pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
    cv.polylines(image, [pts], True, color, 3)


def length(line):
    x1, y1, x2, y2 = line
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def sort_lines_by_length(lines):
    lines_sorted = list(lines)
    lines_sorted.sort(key=distance, reverse=True)
    return lines_sorted


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def longest_line(lines):
    longest = 0
    index = None
    i = 0
    for line in lines:
        x1, y1, x2, y2 = line
        length = distance((x1,y1), (x2,y2))
        if length > longest:
            longest = length
            index = i
        i += 1

    return longest, index


def px_mm_ratio(orig_img, ref_length_in_mm):
    img = orig_img

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # assuming the ref obj is yellow
    # low_yellow = np.array([18, 100, 50])
    # up_yellow = np.array([48, 255, 255])
    low_yellow = np.array([20, 120, 150])
    up_yellow = np.array([48, 255, 255])
    mask = cv.inRange(hsv, low_yellow, up_yellow)
    res = cv.bitwise_and(img, img, mask=mask)
    save_image(res, "yellowMask")

    lines = lines_from_img(res)
    longest, index = longest_line(lines)
    
    if index != None:
        draw_line(img, lines[index], (255,0,0))
        save_image(img, "yellowRefLine")
        return ref_length_in_mm / longest
    return 0


def save_image(img, name):
    cv.imwrite(name+'.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 90])