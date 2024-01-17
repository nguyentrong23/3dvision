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
        return cropped_image,large_image,x,y
    except:
        if not coordinates_str:
            cropped_image= large_image
            return  cropped_image,large_image,0,0
        print("coordinates erro")
        return 0,0,0,0

def preprocess_and_highlight_edges(image,x,y,accept_thresh,accept_angel_low,accept_angel_high,flag,invert =0):
    method = cv2.THRESH_BINARY_INV
    image_shape = (image.shape[0], image.shape[1])
    if invert ==1:
        method = cv2.THRESH_BINARY
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,120, 255,method+cv2.THRESH_OTSU)
    edges = cv2.Canny(thresholded_image, 50, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=150)

    cv2.imshow('thresholded_image', edges)

    list3 = []
    re = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            cv2.line(sr, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
            if accept_angel_low <= angle <= accept_angel_high:
                start, end, length = get_line_endpoints(x1, y1, x2, y2,image_shape)
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                fulline = start[0],start[1],end[0],end[1]
                # print( fulline)
                # print( line)
                # print("2:",line_length)
                persen = (line_length/length)*100
                x1 = x1 + x
                x2 = x2 + x
                y1 = y1 + y
                y2 = y2 + y
                line[0] = x1, y1, x2, y2
                if persen >= accept_thresh:
                    cv2.line(sr, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
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


def get_line_endpoints(x1, y1, x2, y2, image_shape):
    if x1 == x2:
        # print("x")
        start = (x1, 0)
        end = (x1, image_shape[0] - 1)
        length = image_shape[0] - 1
        return start, end, length
    if y1 == y2:
        # print("y")
        start = (0, y1)
        end = (image_shape[1] - 1, y1)
        length = image_shape[1] - 1
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
    x_end1 = image_shape[1] - 1
    y_end1 = int(a * x_end1 + b)
    y_end2 = image_shape[0] - 1
    x_end2 = (y_end2 - b) / a
    if y_end1 <= image_shape[0]:
        end = (x_end1, y_end1)
    elif x_end2 <= image_shape[1]:
        end = (x_end2, y_end2)
    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)   #bug
    return start, end, length

large_image_path = r"datafornichi/protect/1.bmp"
coordinates_str = "1256,31,1390,24,1263,156,1401,166"
sr0,sr,xtop_left,ytop_left = crop_and_process_large_image(large_image_path, coordinates_str)
resutl=preprocess_and_highlight_edges(sr0,xtop_left,ytop_left,1,-180,180,1,invert=1)

for r in resutl:
    x1, y1, x2, y2 = r
    print(x1, y1, x2, y2)
    # cv2.line(sr, (x1, y1), (x2, y2), (0,0,255),2,cv2.LINE_AA)
# sr = cv2.pyrDown(sr)
cv2.imshow('ThresholdedImage',sr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def main():
#     if  len(sys.argv) < 8:
#         print("missing path or thresh")
#     elif len(sys.argv) == 8:
#         path = sys.argv[1]
#         coordinates_str = sys.argv[2]
#         accept_thresh = int(sys.argv[3])
#         min_angel = int(sys.argv[4])
#         max_angel = int(sys.argv[5])
#         flag = int(sys.argv[6])
#         inv = int(sys.argv[7])
#         show = False
#     elif len(sys.argv) > 8:
#         path = sys.argv[1]
#         coordinates_str = sys.argv[2]
#         accept_thresh = int(sys.argv[3])
#         min_angel = int(sys.argv[4])
#         max_angel = int(sys.argv[5])
#         flag = int(sys.argv[6])
#         inv = int(sys.argv[7])
#         show =True
#     try:
#         sr0, sr, xtop_left, ytop_left = crop_and_process_large_image(path, coordinates_str)
#         start = time.time()
#         resutl=preprocess_and_highlight_edges(sr0,xtop_left,ytop_left,accept_thresh,min_angel,max_angel,flag,invert=inv)
#         end = time.time()
#         for r in resutl:
#             x1, y1, x2, y2 = r
#             print("Data: ",x1, y1, x2, y2)
#             cv2.line(sr, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
#         if(show == True):
#             cv2.imshow('ThresholdedImage', sr)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         print("Time : ",end - start)
#         print("Code: 0")
#     except Exception as e:
#         print(f"Msg: {e}")
#         print("Code: -1")
#
# if __name__ == "__main__":
#     main()