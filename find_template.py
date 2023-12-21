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

def fit_angel_pca(contours, hierarchy, src):
    min = 3000
    output = {}
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
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            output[tuple(mean.flatten())] = angel
            scale = 100
            vector2_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            cv2.line(src, mean_point, vector2_end, (0, 255, 0), 1, cv2.LINE_AA)
            text = f"(angel: {angel})"
            cv2.putText(src, text, (int(round(mean[0][0]) - 30), int(round(mean[0][1]) + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # cv2.imshow("detect sro",src)
            # cv2.waitKey(0)
    return output, src

def tienxuly(sr0):
    img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _, edges_src = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy, edges_src

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

path_src = "datafornichi/src/realsense/h11.bmp"
sr0 = cv2.imread(path_src)

sr0 = remove_jig(sr0)
contours, hierarchy_src,edges_src = tienxuly(sr0)

path_tem = "datafornichi/template/h1_tem.bmp"
sr1 = cv2.imread(path_tem)
sr1 = remove_jig(sr1)
contourt, hierarchy_tem,edges_tem = tienxuly(sr1)

template, template_show = fit_angel_pca(contourt, hierarchy_tem, sr1)
object, src_show = fit_angel_pca(contours, hierarchy_src, sr0)
# cv2.imshow(f"src_show",template_show)


# # init parameter
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = edges_tem.shape[:2]
topleft = [0, 0]
y_size = max(h, w)
x_size = edges_src.shape[1]
angel_target = []
mean_target = []

for angle_t in template.values():
    for index,(mean, angles) in enumerate(object.items()):
        angle = angles-angle_t
        # angle = 0
        # Tính toán vị trí góc trái của ROI
        roi_x = 0
        roi_y = int(mean[1] - y_size/2-1)
        roi_y = max(roi_y,0)
        # Tạo ROI (Region of Interest)
        print(roi_y)
        roi = edges_src[roi_y:roi_y + y_size, roi_x:x_size]
        # cv2.imshow(f"detect {angle}", roi)
        # cv2.imshow(f"detec {angle}", edges_tem)
        # cv2.waitKey(0)
        center_rotate = (mean[0], mean[1])
        rotated_src = imutils.rotate(roi, angle, center_rotate)
        res = cv2.matchTemplate(rotated_src, edges_tem, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= 0.9:
            print(max_val,mean, angle)
            mean_target.append(mean)
            angel_target.append(angles)
            topleft = max_loc
            topleft = (topleft[0],topleft[1] + roi_y)
            bottomright = (topleft[0] + w, topleft[1]+h)
            cv2.rectangle(sr0, topleft, bottomright, (0, 255, 255), 1)

            center_x = (topleft[0] + bottomright[0]) // 2
            center_y = (topleft[1] + bottomright[1]) // 2
            cv2.circle(sr0, (center_x, center_y), 3, (0, 0, 255), -1)
        else:
            continue


for index, point in enumerate(mean_target):
    mean_point = (int(round(point[0])), int(round(point[1])))
    cv2.circle(sr0, mean_point, 5, (0, 255, 255), -1)
    print(mean_point)
    text = f"(position: {mean_point[0]}, {mean_point[1]}, index: {index})"
    cv2.putText(sr0, text, (mean_point[0] - 30, mean_point[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
# sr0 = cv2.pyrDown(sr0)
cv2.imshow("detect sro", sr0)
cv2.imshow("detect sr", edges_tem)
cv2.waitKey(0)
cv2.destroyAllWindows()
