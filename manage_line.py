import cv2
import numpy as np
import math
import sys
import time

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
        return cropped_image,x,y,height,width
    except:
        if not coordinates_str:
            cropped_image= large_image
            return  cropped_image,0,0,large_image.shape[0], large_image.shape[1]
        print("coordinates erro")
        return 0,0,0,0,0,0


def fit(sr,invert):
    min = 3000
    # blurred_image = cv2.GaussianBlur(sr, (3, 3), 0)
    img_src = cv2.cvtColor( sr, cv2.COLOR_BGR2GRAY )
    method = cv2.THRESH_BINARY_INV
    if invert ==1:
        method = cv2.THRESH_BINARY
    _,src= cv2.threshold(img_src,120, 255,method+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    result = []
    for index, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= min:
            gradient_angle = np.zeros_like(img_src, dtype=np.uint8)
            cv2.drawContours(gradient_angle, [cnt], 0, (255, 255, 255), 1)
            lines = cv2.HoughLinesP( gradient_angle, 1, np.pi /180, threshold=10, minLineLength=10, maxLineGap=30)
            result.append(lines)
    return result

def preprocess_and_highlight_edges(lines,x,y,w,h,accept_thresh,accept_angel_low,accept_angel_high,flag):
    list3 = []
    re = []
    # thêm canny vao
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if accept_angel_low <= angle <= accept_angel_high:
                start, end, length = get_line_endpoints(x1, y1, x2, y2,w,h)
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                fulline = start[0],start[1],end[0],end[1]
                persen = (line_length/length)*100
                x1 = x1 + x
                x2 = x2 + x
                y1 = y1 + y
                y2 = y2 + y
                line[0] = x1, y1, x2, y2
                if persen >= accept_thresh:
                    cv2.line(forshow, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                    temp = {'line': line, 'fulline': fulline, 'persen': persen,"angel":angle}
                    list3.append(temp)
    if not bool(list3):
        return re
    vertical_lines = []
    horizontal_lines = []
    for match_line in list3:
        angle = match_line["angel"]
        if 45 <= angle <= 135 or -135 <= angle <= -45:
            vertical_lines.append(match_line)
        else:
            horizontal_lines.append(match_line)

    if(flag == 1):
        if not vertical_lines:
            return re
        sorted_list3 = sorted(vertical_lines, key=lambda item: item['persen'], reverse=True)
        last_item = sorted_list3[0]
        re.append(last_item['line'][0])
    elif(flag == 2):
        if not horizontal_lines:
            return re
        sorted_list3 = sorted(horizontal_lines, key=lambda item: item['persen'], reverse=True)
        last_item = sorted_list3[0]
        re.append(last_item['line'][0])
    elif(flag ==3):
        if not horizontal_lines:
            return re
        sorted_list3 = sorted(horizontal_lines, key=lambda item: item['line'][0][1], reverse=True)
        if sorted_list3 is not None:
            # print(sorted_list3[0])
            last_item = sorted_list3[-1]
            re.append(last_item['line'][0])
    elif(flag ==4):
        if not horizontal_lines:
            return re
        sorted_list3 = sorted(horizontal_lines, key=lambda item: item['line'][0][1], reverse=True)
        if sorted_list3 is not None:
            # print(sorted_list3[0])
            last_item = sorted_list3[0]
            re.append(last_item['line'][0])

    elif(flag ==5):
        if not vertical_lines:
            return re
        sorted_list3 = sorted(vertical_lines, key=lambda item: item['line'][0][0], reverse=True)
        if sorted_list3 is not None:
            # print(sorted_list3[0])
            last_item = sorted_list3[-1]
            re.append(last_item['line'][0])
    elif(flag ==6):
        if not vertical_lines:
            return re
        sorted_list3 = sorted(vertical_lines, key=lambda item: item['line'][0][0], reverse=True)
        if sorted_list3 is not None:
            # print(sorted_list3[0])
            last_item = sorted_list3[0]
            re.append(last_item['line'][0])
    else:
        if not vertical_lines:
            return re
        sorted_list3 = sorted(vertical_lines, key=lambda item: item['persen'], reverse=True)
        if sorted_list3 is not None:
            last_item = sorted_list3[0]
            # print(sorted_list3[0])
            # print(sorted_list3)
            re.append(last_item['line'][0])
    return re


def get_line_endpoints(x1, y1, x2, y2, w,h):
    if x1 == x2:
        start = (x1, 0)
        end = (x1, h - 1)
        length = h - 1
        return start, end, length
    if y1 == y2:
        start = (0, y1)
        end = (w - 1, y1)
        length = w - 1
        return start, end, length
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x_1 = 0
    y_1 = b
    y_2 = 0
    x_2 = -b/a
    if y_1 >= 0:
        start = (x_1, y_1)
    elif x_2 >= 0:
        start = (x_2, y_2)
    x_end1 = w - 1
    y_end1 = int(a * x_end1 + b)
    y_end2 = h - 1
    x_end2 = (y_end2 - b) / a
    if y_end1 <= h:
        end = (x_end1, y_end1)
    elif x_end2 <= w:
        end = (x_end2, y_end2)
    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return start, end, length


path = r"datafornichi/protect/1.bmp"
coordinates ="123,13,1574,12,121,208,1575,206"
start_time = time.time()
sr0,xtl,ytl,h,w = crop_and_process_large_image(path,coordinates)
line_list = fit(sr0,0)
forshow = cv2.imread(path)
for object_line in line_list:
    result = preprocess_and_highlight_edges(object_line, xtl,ytl,w,h,1,-100,100,4)
    for r in  result:
        x1, y1, x2, y2 = r
        print(x1, y1, x2, y2)
        cv2.line(forshow, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    # for line in object_line:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(sr0, (x1, y1), (x2, y2), (0, 255, 0),1,cv2.LINE_AA)
    cv2.imshow('Original Image', forshow)
    cv2.waitKey(0)

cv2.destroyAllWindows()