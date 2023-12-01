import os
import cv2
import numpy as np
import math
import time
try :
    import imutils
except:
    os.system("pip install  imutils")
    import imutils
#  code mang tính đặt thù chỉ chính  xác vật thể cho trước

def fit_angel_pca(contours, src):
    min = 3500
    angel_output = []
    mean_output = []
    for index, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        if area >= min and area <=6000 :
            print(area)
            cv2.drawContours(src, [cnt], -1, (0,0, 0),1, cv2.LINE_AA)
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            angel_output.append(angel)
            mean_output.append(mean)
            scale = 100
            vector_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            cv2.arrowedLine(src, mean_point, vector_end, (0, 255, 0), 1)
    return mean_output,angel_output, src


# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("template_pipe2.jpg")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3,3), 0)
_, edges_src = cv2.threshold(blurred,80, 120, cv2.THRESH_BINARY_INV)
contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


# # Đọc ảnh và tiền xử lý source
# sr0 = cv2.imread("template_pipe2.jpg")
#
# #  giảm bóng
# hsv_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2HSV)
# brightness_factor = 2.5
# hsv_src[:, :, 2] = np.clip(hsv_src[:, :, 2] * brightness_factor, 0, 255)
# brg_src = cv2.cvtColor(hsv_src, cv2.COLOR_HSV2BGR)
# blurred_src = cv2.bilateralFilter(brg_src , d=3, sigmaColor=75, sigmaSpace=5)
#
# # chuyển xám
# gray_src = cv2.cvtColor(blurred_src,cv2.COLOR_BGR2GRAY)
# _, edges_src = cv2.threshold(gray_src,140, 150, cv2.THRESH_BINARY_INV)
# contours_src, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


# #  đọc  và tiền xử lý template
sr1= cv2.imread("template.jpg")
#  giảm bóng
hsv_tem = cv2.cvtColor(sr1, cv2.COLOR_BGR2HSV)
brightness_factor = 2.5
hsv_tem[:, :, 2] = np.clip(hsv_tem[:, :, 2] * brightness_factor, 0, 255)
brg_tem = cv2.cvtColor(hsv_tem, cv2.COLOR_HSV2BGR)
blurred_tem = cv2.bilateralFilter(brg_tem, d=3, sigmaColor=75, sigmaSpace=5)
# chuyển xám
gray_template = cv2.cvtColor(blurred_tem,cv2.COLOR_BGR2GRAY)
_, edges_template = cv2.threshold(gray_template,140, 150, cv2.THRESH_BINARY_INV)
contours_temp, hierarchy_temp = cv2.findContours(edges_template, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#
# # init parameter
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = gray_template.shape[::]
topleft = [0,0]
best_angel = 0
src_copy = np.zeros_like(img_src)
angel_taget = []
mean_taget = []

# # resolve angel problem by pca
#
# center_temp,angel_temp,  template_show = fit_angel_pca(contours_temp,sr1)
# cv2.imshow("detect sro",template_show)


start_time = time.time()
center_cons,angel_src, src_show = fit_angel_pca(contours,sr0)
cv2.imshow(" src_show", src_show)
#
# for idx,angels in enumerate(angel_src):
#     angel = angels - angel_temp[0]
#     center_rotate = (int(round(center_cons[idx][0][0])), int(round(center_cons[idx][0][1])))
#     rotated_src = imutils.rotate(edges_src,angel,center_rotate)
#     res = cv2.matchTemplate(rotated_src,edges_template, method)
#     minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
#     print(maxval)
#     cv2.imshow(f"detect {idx}",  rotated_src)
#     if(maxval>=0.75):
#         mean_taget.append(center_cons[idx])
#         angel_taget.append(angels)
#
#     else:
#             continue
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Thời gian chạy: {execution_time} giây")
#
# for index,point in enumerate(mean_taget):
#     mean_point = (int(round(point[0][0])), int(round(point[0][1])))
#     cv2.circle(sr0, mean_point, 5, (0, 255, 255), -1)
#     note = str(angel_taget[index])
#     cv2.putText(sr0, note , mean_point, cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1)
# cv2.imshow("detect sro",sr0)
# cv2.imshow("edges_temlate",edges_template)

cv2.waitKey(0)
cv2. destroyAllWindows

