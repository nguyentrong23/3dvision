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
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,0, 255,method + cv2.THRESH_OTSU)

    edges = cv2.Canny(thresholded_image, 50, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=30)
    list3 = []
    re = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 = x1+x
            x2 = x2+x
            y1 = y1+y
            y2 = y2+y
            line[0]=x1, y1, x2, y2
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if accept_angel_low <= angle <= accept_angel_high:
                start, end, length = get_line_endpoints(x1, y1, x2, y2,image_shape)
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                fulline = start[0],start[1],end[0],end[1]
                persen = (line_length/length)*100
                if persen >= accept_thresh:
                    temp = {'line': line, 'fulline': fulline, 'persen': persen,"angel":angle}
                    list3.append(temp)
    if(flag == 1):
        sorted_list3 = sorted(list3, key=lambda item: item['persen'], reverse=True)
        if sorted_list3 is not None:
            last_item = sorted_list3[0]
            # print(last_item['persen'])
            # print(last_item['line'][0])
            re.append(last_item['line'][0])
    elif(flag ==2):
        sorted_list3 = sorted(list3, key=lambda item: item['line'][0][1], reverse=True)
        if sorted_list3 is not None:
            last_item = sorted_list3[-1]
            # print(last_item['line'][0])
            re.append(last_item['line'][0])

    elif(flag ==3):
        sorted_list3 = sorted(list3, key=lambda item: item['line'][0][1], reverse=True)
        if sorted_list3 is not None:
            last_item = sorted_list3[0]
            # print(last_item['line'][0])
            re.append(last_item['line'][0])
    else:
        sorted_list3 = sorted(list3, key=lambda item: item['persen'], reverse=True)
        if sorted_list3 is not None:
            last_item = sorted_list3[0]
            # print(last_item['persen'])
            # print(last_item['line'][0])
            re.append(last_item['line'][0])
    return re


def get_line_endpoints(x1, y1, x2, y2, image_shape):
    a = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
    b = y1 - a * x1
    x_1 = 0
    y_1 = int(a * x_1 + b)
    y_2 = 0
    try:
        x_2 =-b / a if a != 0 else np.inf
    except ZeroDivisionError:
        x_2 = -1
    if y_1 >= 0:
        start = (x_1, y_1)
    elif x_1 >= 0:
        start = (x_2, y_2)
    x_end1 = image_shape[1] - 1
    y_end1 = int(a * x_end1 + b)
    y_end2 = image_shape[0] - 1

    try:
        x_end2 = (y_end2 - b) / a if a != 0 else np.inf  # Tránh chia cho 0
    except ZeroDivisionError:
        x_end2 = -1

    if y_end1 <= image_shape[0]:
        end = (x_end1, y_end1)
    elif x_1 <= image_shape[1]:
        end = (x_end2, y_end2)
    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    return start, end, length


large_image_path = r"datafornichi\template\h11_tem.bmp"
coordinates_str = "134,1,434,1,134,180,434,180"
sr0,sr,xtop_left,ytop_left = crop_and_process_large_image(large_image_path, coordinates_str)
resutl=preprocess_and_highlight_edges(sr0,xtop_left,ytop_left,5,-10,10,1,invert=1)

for r in resutl:
    x1, y1, x2, y2 = r
    print(x1, y1, x2, y2)
    cv2.line(sr, (x1, y1), (x2, y2), (0,0,255), 1,cv2.LINE_AA)
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
#     try:
#         sr0, sr, xtop_left, ytop_left = crop_and_process_large_image(path, coordinates_str)
#         start = time.time()
#         resutl=preprocess_and_highlight_edges(sr0,xtop_left,ytop_left,accept_thresh,min_angel,max_angel,flag,invert=inv)
#         end = time.time()
#         for r in resutl:
#             x1, y1, x2, y2 = r
#             print("Data: ",x1, y1, x2, y2)
#             cv2.line(sr, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
#             cv2.imshow('ThresholdedImage', sr)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         print("Time : ",end - start)
#         print("Code: 0")
#     except Exception as e:
#         print(f"Msg: {e}")
#         print("Code: -1")
#
# if __name__ == "__main__":
#     main()