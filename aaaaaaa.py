import os
import cv2
import numpy as np
import math
import time
try:
    import imutils
except:
    os.system("pip install imutils")
    import imutils

def fit_angel_pca(sr):
    min = 3000
    output = {}
    # tien xu ly
    img_src = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _,src = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        # if hierarchy[0, index, 2] != -1:
        #     continue;
        area = cv2.contourArea(cnt)
        if area >= min:
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(sr, mean_point, 5, (0, 0, 255), -1)
            output[tuple(mean.flatten())] = angel
            scale = 100
            vector2_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            cv2.line(sr, mean_point, vector2_end, (0, 255, 0), 1, cv2.LINE_AA)
            text = f"(angel: {angel})"
            cv2.putText(sr, text, (int(round(mean[0][0]) - 30), int(round(mean[0][1]) + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.imshow('Original Image', sr)
    return output,src,contours,hierarchy

def remove_jig(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    # Tạo mask để chỉ giữ lại các pixel nằm trong khoảng giá trị màu xanh dương
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[(hsv[:, :, 0] >= lower_blue[0]) & (hsv[:, :, 0] <= upper_blue[0]) &
         (hsv[:, :, 1] >= lower_blue[1]) & (hsv[:, :, 1] <= upper_blue[1]) &
         (hsv[:, :, 2] >= lower_blue[2]) & (hsv[:, :, 2] <= upper_blue[2])] =255
    result = cv2.bitwise_or(img, mask)
    # cv2.imshow('Original Image', mask)
    # cv2.imshow('Image without Blue-Green Objects', result)
    return result

def padding(img,size):
    top, bottom, left, right = size, size,0,0
    border_color = [0, 0, 0]
    image_with_padding = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
    return image_with_padding

def rotate_image(image, angle,center):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_point_in_image(image, point, angle_degrees):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 1]))
    new_x, new_y = rotated_point[:2].astype(int)
    newpoint = (int(new_x),int(new_y))
    return newpoint



def matching(edges_src,edges_tem,template,object,min_thresh,sr0):
    # # init parameter
    method = eval("cv2.TM_CCOEFF_NORMED")
    h, w = edges_tem.shape[:2]
    topleft = [0, 0]
    y_size = max(h, w)
    x_size = edges_src.shape[1]
    angel_target = []
    mean_target = []
    for angle_t in template.values():
        for index, (mean, angles) in enumerate(object.items()):
            # angle = angles - angle_t
            angle = angles
            rotated_src = imutils.rotate(edges_src,angle)
            roi_x = 0
            roi_y = int(mean[1] - y_size/2)
            roi_y = max(roi_y, 0)
            # Tạo ROI (Region of Interest)
            roi = rotated_src[roi_y:roi_y + y_size, roi_x:x_size]
            cv2.imshow('roi', roi)
            if(roi_y+y_size) > edges_src.shape[0]:
                size = roi_y+y_size - edges_src.shape[0]
                roi =  padding(roi, size)
                roi_y = roi_y - size
            res = cv2.matchTemplate(roi, edges_tem, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # print(max_val)
            if max_val >= min_thresh:

                # sr0 = imutils.rotate(sr0, angle)
                topleft = max_loc
                topleft = (topleft[0], topleft[1] + roi_y)
                topleft = rotate_point_in_image(rotated_src,topleft, -angle)
                bottomright = (topleft[0] + w, topleft[1] + h)
                cv2.rectangle(sr0, topleft, bottomright, (0, 255, 255), 1)
                center_x = (topleft[0] + bottomright[0]) // 2
                center_y = (topleft[1] + bottomright[1]) // 2
                m_target = (center_x,center_y)
                cv2.circle(sr0,(m_target[0],m_target[1]), 3, (0, 255, 255), -1)
                # sr0 = imutils.rotate(sr0, -angle)
                mean_target.append(m_target)
                angel_target.append(angles)

                # cv2.imshow('show_src', sr0)
                # cv2.waitKey(0)

    # sr0 = cv2.pyrDown(sr0)
    # sr0 = cv2.pyrDown(sr0)
    cv2.imshow('Original Image', sr0)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time} giây")
    cv2.waitKey(0)
    return  angel_target,mean_target

path_src = "datafornichi/src/realsense/h11.bmp"
sr0 = cv2.imread(path_src)

path_tem = "datafornichi/src/realsense/h11_tem.bmp"
sr1 = cv2.imread(path_tem)

sr1 = remove_jig(sr1)
sr0 = remove_jig(sr0)


start_time = time.time()

template,edges_tem,contourt,_ = fit_angel_pca(sr1)
object,edges_src,contours,hierarchy = fit_angel_pca(sr0)

angel, mean = matching(edges_src,edges_tem,template,object,0.5,sr0)

print(angel,mean)

cv2.waitKey(0)
cv2.destroyAllWindows()
