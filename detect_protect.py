import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
import time
import sys
try:
    import imutils
except:
    os.system("pip install imutils")
    import imutils

def fit4ROI(sr):
    min = 3000
    output = []
    show = []
    rang_coor = []
    method = cv2.THRESH_BINARY_INV
    img_src = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _, thresholded_image = cv2.threshold(img_src,120, 255,method+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        area = cv2.contourArea(cnt)
        if area >= min:
            data = cnt[:, 0, :].astype(np.float32)
            # Get the highest and lowest y-coordinates
            min_y = int(np.min(data[:, 1])) -20
            max_y = int(np.max(data[:, 1])) +20
            roi = img_src[min_y:max_y, :]
            roi_show = sr[min_y:max_y, :]
            output.append(roi)
            show.append( roi_show)
            rang_coor.append(min_y)
    return output,show,rang_coor


def crop_and_process_large_image(large_image_path, coordinates_str):
    large_image = cv2.imread(large_image_path)
    try:
        coordinates = list(map(int, coordinates_str.split(',')))
        x1, y1, x2, y2, x4, y4, x3, y3 = coordinates
        x = int(min(x1, x2, x3, x4))-5
        y = int(min(y1, y2, y3, y4))-5
        width = int(max(x1, x2, x3, x4) - x)+5
        height = int(max(y1, y2, y3, y4) - y)+5
        cropped_image = large_image[y:y + height, x:x + width]
        return cropped_image
    except:
        if not coordinates_str:
            cropped_image= large_image
            # cv2.imshow('cropped_image', cropped_image)
            return  cropped_image
        print("coordinates erro")
        return 0

def get_gradient_sobel(image):
    _, edges_src = cv2.threshold(image,120,255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    edges = cv2.Canny(edges_src, 20, 160)
    contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(edges_src, dtype=np.uint8)
    for cons in contours:
        area = cv2.contourArea(cons)
        if area > 1000:
            cv2.drawContours(mask, [cons], -1, (255),thickness=cv2.FILLED)

    sobel_x = cv2.Sobel(edges_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edges_src, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))
    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[(direc_angle >=-100) & (direc_angle <=-80)] = 255
    gradient_angle_flip[(direc_angle >= 80) & (direc_angle <= 100)] = 255
    mask = cv2.bitwise_not(mask)
    gradient_angle = cv2.bitwise_and(gradient_angle, mask)
    gradient_angle_flip = cv2.bitwise_and(gradient_angle_flip, mask)

    data_bottom = np.where(gradient_angle != 0)
    point_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(gradient_angle_flip != 0)
    point_top = np.column_stack((data_top[1], data_top[0]))
    cv2.imshow('gradient_angle_flip', gradient_angle_flip)
    return edges,  point_top , point_bottom


def group_points_by_y(points, y_threshold=2):
    groups = []
    current_group = []
    sorted_points = sorted(points, key=lambda x: x[1])
    for i in range(len(sorted_points) - 1):
        current_point = sorted_points[i]
        next_point = sorted_points[i + 1]
        current_group.append(current_point)
        if abs(next_point[1] - current_point[1]) > y_threshold:
            current_group = sorted(current_group, key=lambda x: x[0])
            groups.append(current_group)
            current_group = []
    current_group.append(sorted_points[-1])
    current_group = sorted(current_group, key=lambda x: x[0])
    groups.append(current_group)
    return groups

def find_edges_inROI(listROI,show):
    for idx,ob in enumerate(listROI):
        e, bot, top = get_gradient_sobel(ob)
        line_bot = group_points_by_y(bot)
        line_top = group_points_by_y(top)
        for i,group in enumerate(line_bot):
                # print(group[-1])
                # print(group[0])
                point_start = (group[0][0],group[0][1])
                point_end = (group[-1][0],group[-1][1])
                cv2.circle(show[idx],point_start, 3, (0, 0, 255), -1)
                cv2.circle(show[idx],point_end, 3, (0, 0, 255), -1)
                # for point in group:
                #     print(point,i)
        for i,group in enumerate(line_top):
                # print(group[-1])
                # print(group[0])
                point_start = (group[0][0],group[0][1])
                point_end = (group[-1][0],group[-1][1])
                cv2.circle(show[idx],point_start, 3, (0, 0, 255), -1)
                cv2.circle(show[idx],point_end, 3, (0, 0, 255), -1)
                # for point in group:
                #     print(point,i)
        cv2.imshow('3333333333333',show[idx])
        cv2.waitKey(0)

large_image_path = r"datafornichi/protect/1.bmp"
coordinates_str = "153,51,298,48,156,782,289,798"

sr0 = crop_and_process_large_image(large_image_path, coordinates_str)
listROI,list_show,topleft = fit4ROI(sr0)
find_edges_inROI(listROI,list_show)

cv2.imshow('3333333333333',sr0)
cv2.waitKey(0)


