import os
import cv2
import numpy as np
import math
import time
import sys

def count_obj(sr):
    min = 10000
    output = []
    img_src = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _,src = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    count =0
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        area = cv2.contourArea(cnt)
        if area >= min:
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            output.append(mean_point)
            cv2.circle(sr, mean_point, 5, (0, 0, 255), -1)
            count +=1
    return output,count

def remove_jig(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([40, 40, 40])
    upper_blue = np.array([200, 255, 255])
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[(hsv[:, :, 0] >= lower_blue[0]) & (hsv[:, :, 0] <= upper_blue[0]) &
         (hsv[:, :, 1] >= lower_blue[1]) & (hsv[:, :, 1] <= upper_blue[1]) &
         (hsv[:, :, 2] >= lower_blue[2]) & (hsv[:, :, 2] <= upper_blue[2])] =255
    result = cv2.bitwise_or(img, mask)
    return result

def crop_mid(large_image_path):
    large_image = cv2.imread(large_image_path)
    try:
        h,w = large_image.shape[0:2]
        left = int((w/5)*2)
        right = int(w-left)
        cropped_image = large_image[:,left:right]
        return cropped_image
    except:
        print("coordinates erro")
        return 0
def deb():
    path_src = r"datafornichi/ttttttt/3.bmp"
    sr0 =crop_mid(path_src)
    # sr0 = remove_jig(sr0)
    coor,count = count_obj(sr0)
    out_string = ""
    for i in reversed(coor):
        note = str(i[1]) + ","
        out_string += note

    print("count: ", count)
    print("coor: ", out_string)
    cv2.imshow('Original Image', sr0 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if  len(sys.argv) < 2:
        print("missing path:")
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        show = False
    elif len(sys.argv) > 2:
        path = sys.argv[1]
        show = True
    try:
        sr0 =crop_mid(path)
        # sr0 = remove_jig(sr0)
        start_time = time.time()
        coor,count = count_obj(sr0)
        end_time = time.time()
        out_string = ""
        for i in reversed(coor):
            note = str(i[1]) + ","
            out_string += note
        print("Code: 0")
        print(f"data:{count}/{out_string}")
        # print(f"count: {count}")
        # print(f"coor: {out_string}")
        print(f"time: {start_time-end_time}")
        if show:
            cv2.imshow('Original Image', sr0 )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
            print("Code: -1")
            print(f"Msg: {e}")

if __name__ == "__main__":
    main()
