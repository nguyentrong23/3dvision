import cv2
import numpy as np
import math
import sys

def get_gradient_sobel(image,low,high):
    image = increase_contrast(image)
    blurred = cv2.pyrMeanShiftFiltering(image, low, high)
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    cv2.imshow('0', gray_image)
    kernel = np.ones((3, 3), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('2', gray_image)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))
    cv2.imshow('direc_angle', direc_angle)
    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[direc_angle == 90] = 255
    gradient_angle_flip[direc_angle == -90] = 255

    cv2.imshow('direc_angle', direc_angle)
    cv2.imshow(' gradient_angle',  gradient_angle)
    # cv2.imshow('gradient_angle_flip', gradient_angle_flip)
    _, binary_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary_image, 100, 200)
    cv2.imshow('3',edges)
    point_top = cv2.bitwise_and(gradient_angle, edges)
    point_bottom = cv2.bitwise_and(gradient_angle_flip, edges)
    data_bottom = np.where(point_top != 0)
    data_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(point_bottom != 0)
    data_top = np.column_stack((data_top[1], data_top[0]))
    lines_top = cv2.HoughLinesP(point_bottom, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)
    lines_bot = cv2.HoughLinesP(point_top, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)
    try:
        for line in lines_top:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            print(x1, y1, x2, y2)
        for line in lines_bot:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
            print(x1, y1, x2, y2)
    except:
        print("noline")
    cv2.imshow('00', point_top)
    cv2.imshow('01', point_bottom)

    cv2.imshow('1',image)

    return edges, data_top, data_bottom

def increase_contrast(image, alpha=1.5, beta=5):
    # Áp dụng phép biến đổi pixel để tăng độ tương phản
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return  adjusted_image


# Đọc ảnh
image = cv2.imread("datafornichi/NG001_lite.jpg")
edges, TopLine, Botline = get_gradient_sobel(image,60,120)
cv2.waitKey(0)
cv2.destroyAllWindows()