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

def fit_angel_pca(contours, hierarchy, src):
    min = 1000
    angel_output = []
    mean_output = []
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        if hierarchy[0, index, 2] != -1:
            continue;
        area = cv2.contourArea(cnt)
        if area >= min and area <=15000:
            cv2.drawContours(src, [cnt], -1, (0,0, 0),1, cv2.LINE_AA)
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            angel_output.append(angel)
            mean_output.append(mean)
            scale = 100
            vector2_end = (int(mean_point[0] + eigenvectors[1][0] * scale), int(mean_point[1] + eigenvectors[1][1] * scale))
            cv2.arrowedLine(src, mean_point, vector2_end, (0, 255, 0), 2)
    return mean_output,angel_output, src




# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("data\Test Images\Src9.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3,3), 0)
_, edges_src = cv2.threshold(blurred,80, 120, cv2.THRESH_BINARY_INV)
contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


#  đọc  và tiền xử lý template
sr1= cv2.imread("data/Test Images/Dst9.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
_, edges_temlate = cv2.threshold(blurred, 80, 120, cv2.THRESH_BINARY_INV)
contours_temp, hierarchy_temp = cv2.findContours(edges_temlate, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# init parameter
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
topleft = [0,0]
best_angel = 0
src_copy = np.zeros_like(img_src)
angel_taget = []
mean_taget = []

# resolve angel problem by pca

center_temp,angel_temp,  template_show = fit_angel_pca(contours_temp,hierarchy_temp,sr1)
start_time = time.time()
center_cons,angel_src, src_show = fit_angel_pca(contours,hierarchy_src,sr0)

for idx,angels in enumerate(angel_src):
    angel = angels - angel_temp[0]
    center_rotate = (int(round(center_cons[idx][0][0])), int(round(center_cons[idx][0][1])))
    rotated_src = imutils.rotate(edges_src,angel,center_rotate)# xoay quanh tam cua object
    res = cv2.matchTemplate(rotated_src,edges_temlate, method)

    minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
    print(maxval)
    cv2.imshow(f"detect {idx}",  rotated_src)
    if(maxval>=0.7):
        mean_taget.append(center_cons[idx])
        angel_taget.append(angels)

    else:
            continue

end_time = time.time()
execution_time = end_time - start_time
print(f"Thời gian chạy: {execution_time} giây")

for index,point in enumerate(mean_taget):
    mean_point = (int(round(point[0][0])), int(round(point[0][1])))
    cv2.circle(sr0, mean_point, 5, (0, 255, 255), -1)
    note = str(angel_taget[index])
    cv2.putText(sr0, note , mean_point, cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 1)
cv2.imshow("detect sro",sr0)
cv2.imshow("edges_temlate",edges_temlate)

cv2.waitKey(0)
cv2. destroyAllWindows

