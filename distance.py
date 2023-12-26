import cv2
import numpy as np
import math
import sys
import time

def increase_contrast(image, alpha=1.0, beta=1):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return  adjusted_image


def get_gradient_sobel(image,thresh):
    image = increase_contrast(image)
    blurred = cv2.GaussianBlur(image,(5,5), 0)
    blurred = cv2.pyrMeanShiftFiltering(blurred,30,60)
    img_src = cv2.cvtColor(blurred , cv2.COLOR_BGR2GRAY)
    _, edges_src = cv2.threshold(img_src,thresh,255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(edges_src, 80, 160)
    contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(edges_src, dtype=np.uint8)
    for cons in contours:
        area = cv2.contourArea(cons)
        if area > 1000:
            cv2.drawContours(mask, [cons], -1, (255),thickness=cv2.FILLED)
    # sobel
    sobel_x = cv2.Sobel(edges_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edges_src, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))

    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[direc_angle == -90] = 255
    gradient_angle_flip[direc_angle == 90] = 255

    mask = cv2.bitwise_not(mask)
    # #   end với cạnh trên và cạnh dưới để bỏ ruột
    gradient_angle = cv2.bitwise_and(gradient_angle, mask)
    gradient_angle_flip = cv2.bitwise_and(gradient_angle_flip, mask)
    data_bottom = np.where(gradient_angle != 0)
    point_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(gradient_angle_flip != 0)
    point_top = np.column_stack((data_top[1], data_top[0]))
    lines_top = cv2.HoughLinesP(gradient_angle, 1, np.pi / 180,  threshold=15, minLineLength=15, maxLineGap=30)
    lines_bot = cv2.HoughLinesP(gradient_angle_flip, 1, np.pi / 180, threshold=15, minLineLength=15, maxLineGap=30)
    try:
        for line in lines_top:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        for line in lines_bot:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    except:
        print("noline")
    # cv2.imshow('image', image)
    return edges,  lines_top , lines_bot

def crop_and_process_large_image(large_image_path, coordinates_str):
    large_image = cv2.imread(large_image_path)
    try:
        coordinates = list(map(int, coordinates_str.split(',')))
        x1, y1, x2, y2, x4, y4, x3, y3 = coordinates
        x = int(min(x1, x2, x3, x4))-5
        y = int(min(y1, y2, y3, y4))-5
        width = int(max(x1, x2, x3, x4) - x)+5
        height = int(max(y1, y2, y3, y4) - y)+5
        # Crop ảnh lớn để lấy ảnh nhỏ
        cropped_image = large_image[y:y + height, x:x + width]
        # cv2.imshow('Large Image', large_image)
        # cv2.imshow('cropped_image', cropped_image)
        return cropped_image
    except:
        if not coordinates_str:
            cropped_image= large_image
            cv2.imshow('cropped_image', cropped_image)
            return  cropped_image
        print("coordinates erro")
        return 0


large_image_path = "datafornichi/src/realsense/h11.bmp"
coordinates_str = "65,95,150,91,68,725,147,720"
sr0 = crop_and_process_large_image(large_image_path, coordinates_str)
edges, TopLine, Botline = get_gradient_sobel(sr0,180)
print(TopLine)
print(Botline)
for it,gt in enumerate(TopLine):
    for ib,gb in enumerate(Botline):
        if(ib == it):
            x1, y1, x2, y2 = gt[0]
            x3, y3, x4, y4 = gb[0]
            cv2.line(sr0, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(sr0, (x3, y3), (x4, y4), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('333',sr0)
            cv2.waitKey(0)
            #
            # print(f'Khoảng cách giữa hai vector là: {distance}')
            # print(f'độ phân giải ảnh là: {edges.shape[::]}')
# sr0 = cv2.pyrUp(sr0)
cv2.imshow('333',sr0)


# cv2.imwrite("do_khoang_cach_roi.png",sr0)
cv2.waitKey(0)
cv2.destroyAllWindows()







