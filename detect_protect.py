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
            min_y = int(np.min(data[:, 1])) -10
            max_y = int(np.max(data[:, 1])) +10
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
        x = int(min(x1, x2, x3, x4))
        y = int(min(y1, y2, y3, y4))
        width = int(max(x1, x2, x3, x4) - x)
        height = int(max(y1, y2, y3, y4) - y)
        cropped_image = large_image[y:y + height, x:x + width]
        topleft = (x,y)
        return cropped_image,topleft
    except:
        if not coordinates_str:
            cropped_image= large_image
            # cv2.imshow('cropped_image', cropped_image)
            return  cropped_image,(0,0)
        print("coordinates erro")
        return 0,(0,0)

def get_gradient_sobel(image):
    _, edges_src = cv2.threshold(image,120,255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    edges = cv2.Canny(edges_src, 20, 160)
    contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(edges_src, dtype=np.uint8)
    for cons in contours:
        area = cv2.contourArea(cons)
        if area > 1000:
            cv2.drawContours(mask, [cons], -1, (255),thickness=cv2.FILLED)

    sobel_x = cv2.Sobel(edges_src, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(edges_src, cv2.CV_64F, 0, 1, ksize=1)
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
    # cv2.imshow('gradient_angle_flip', gradient_angle_flip)
    # cv2.imshow('gradient_angle', gradient_angle)
    return edges,  point_top , point_bottom


def group_points_by_y(points, y_threshold=5):
    groups = []
    current_group = []
    sorted_points = sorted(points, key=lambda x: (x[0], x[1]))
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

def find_edges_inROI(listROI,show,topleft,flag):
    result = []
    for idx,ob in enumerate(listROI):
        e, bot, top = get_gradient_sobel(ob)
        line_bot = group_points_by_y(bot)
        line_top = group_points_by_y(top)
        tl = []
        tr = []
        bl = []
        br = []
        for i,group in enumerate(line_bot):
                bot_start = (group[0][0],group[0][1])
                bot_end = (group[-1][0],group[-1][1])
                cv2.circle(show[idx],bot_start, 3, (0, 0, 255), -1)
                cv2.circle(show[idx],bot_end, 3, (0, 0, 255), -1)
                br.append(bot_end)
                bl.append(bot_start)
        for i,group in enumerate(line_top):
                top_start = (group[0][0],group[0][1])
                top_end = (group[-1][0],group[-1][1])
                cv2.circle(show[idx],top_start, 3, (0, 0, 255), -1)
                cv2.circle(show[idx],top_end, 3, (0, 0, 255), -1)
                tr.append(top_end)
                tl.append(top_start)
        if(flag ==1):
            cv2.line(show[idx], tr[0], br[0], (255, 255, 0), 1, cv2.LINE_AA)
            point1 = (tr[0][0],tr[0][1] + topleft[idx])
            point2 = (br[0][0],br[0][1] + topleft[idx])
            result.append([point1,point2])
        else:
            cv2.line(show[idx], tl[-1], bl[-1], (255, 255, 0), 1, cv2.LINE_AA)
            point1 = (tl[-1][0],tl[-1][1] + topleft[idx])
            point2 = (bl[-1][0],bl[-1][1] + topleft[idx])
            result.append([point1,point2])
    return result


# large_image_path = r"datafornichi/protect/1.bmp"
large_image_path = r"datafornichi/protect/2.bmp"
coordinates_left = "120,0,384,0,120,784,398,788"
coordinates_right = "1129,0,1369,0,1173,775,1375,788"
# coordinates_left   = "300,0,384,0,300,784,398,788"
# coordinates_right = "1129,0,1200,0,1173,775,1200,784"

sr0,top1 = crop_and_process_large_image(large_image_path,coordinates_left)
listROI,list_show,topleft4l = fit4ROI(sr0)
re1 = find_edges_inROI(listROI,list_show,topleft4l,0)

sr1,top2 = crop_and_process_large_image(large_image_path,coordinates_right)
listROI1,list_show,topleft4r = fit4ROI(sr1)
re2 = find_edges_inROI(listROI1,list_show,topleft4r,1)
image = cv2.imread(large_image_path)


outline1 = []
outline2 = []

for line in re1:
    a,b = top1
    point1x = line[0][0] + a
    point1y = line[0][1] + b
    point2x = line[1][0] + a
    point2y = line[1][1] + b
    point1 = (point1x ,point1y)
    point2 = (point2x ,point2y)
    cv2.line(image,point1,point2, (255, 255, 0), 1, cv2.LINE_AA)
    mean_x = int((point1x + point2x) / 2)
    mean_y = int((point1y +  point2y) / 2)
    mean_point = (mean_x, mean_y)
    cv2.circle(image,mean_point,1, (0, 0, 255), -1)
    outline1.append(mean_point)

for line in re2:
    a,b = top2
    point1x = line[0][0] + a
    point1y = line[0][1] + b
    point2x = line[1][0] + a
    point2y = line[1][1] + b
    point1 = (point1x ,point1y)
    point2 = (point2x ,point2y)
    cv2.line(image,point1,point2, (255, 255, 0), 1, cv2.LINE_AA)
    mean_x = int((point1x + point2x) / 2)
    mean_y = int((point1y +  point2y) / 2)
    mean_point = (mean_x, mean_y)
    cv2.circle(image,mean_point,1, (0, 0, 255), -1)
    outline2.append(mean_point)

for idx,_ in enumerate(outline1):
    cv2.line(image,outline1[idx],outline2[idx], (0,255,255), 1, cv2.LINE_AA)
    distance = math.sqrt((outline2[idx][0]- outline1[idx][0]) ** 2 + (outline2[idx][1] - outline2[idx][1]) ** 2)
    text = f"{distance} pixel"
    ll = outline1[idx]
    rr= outline2[idx]

    mean_x = int((ll[0] + rr[0]) / 2)
    mean_y = int((ll[1] + rr[1]) / 2)
    mean_point = (mean_x, mean_y)
    cv2.putText(image, text, mean_point, cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1)
    print(distance, " pixel")
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
cv2.imshow('image', image)
cv2.imshow('r',sr1)
cv2.imshow('l',sr0)
cv2.waitKey(0)


