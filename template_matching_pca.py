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
    min_area = 6000
    min_arange = 1000
    angel_output = []
    mean_output = []
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        if hierarchy[0, index, 2] == -1:
            continue;
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(src, [cnt], -1, (0,0, 0),1, cv2.LINE_AA)
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            danhsach = len(hierarchy[0])
            for count_her in range(0,danhsach):
                indexof = hierarchy[0, count_her, 3]
                if indexof == index:
                    son_area = cv2.contourArea(contours[count_her])
                    if son_area >= min_arange :
                        data_son = contours[count_her][:, 0, :].astype(np.float32)
                        mean_son, _ = cv2.PCACompute(data_son, mean=None)
                        means_point = (int(round(mean_son[0][0])), int(round(mean_son[0][1])))
                        # cv2.circle(src, means_point, 5, (0, 0, 255), -1)
                        # # Vẽ vector từ mean_point đến means_son
                        # cv2.arrowedLine(src, mean_point, means_point, (0, 0, 0), 1)

                        # Tính góc của vector
                        dx = means_point[0] - mean_point[0]
                        dy = means_point[1] - mean_point[1]
                        angle = math.atan2(dy, dx) * 180 / math.pi
                        angel_output.append(angle)
                        mean_output.append(mean)


    return mean_output,angel_output, src




# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("data\imgSrc\sample.bmp")
img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
_, edges_src = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)
contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


#  đọc  và tiền xử lý template
sr1= cv2.imread("data/imgSrc/template.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
_, edges_temlate = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)
contours_temp, hierarchy_temp = cv2.findContours(edges_temlate, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


#  đọc  và tiền xử lý template flip
sr1_flip= cv2.imread("data/imgSrc/template-flip.bmp")
img_template_flip = cv2.cvtColor(sr1_flip,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template_flip, (3,3), 0)
_,edges_template_flip = cv2.threshold(blurred, 100, 160, cv2.THRESH_BINARY_INV)
contours_temp_flip, hierarchy_flip = cv2.findContours(edges_template_flip, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


# init parameter
# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
topleft = [0,0]
best_angel = 0
src_copy = np.zeros_like(img_src)
angel_taget = []
mean_taget = []


# resolve angel problem by pca

center_temp,angel_temp,  template_show = fit_angel_pca(contours_temp,hierarchy_temp,sr1)
center_temp_flip,angel_flip,  template_show_flip = fit_angel_pca(contours_temp_flip,hierarchy_flip,sr1_flip)

start_time = time.time()
center_cons,angel_src, src_show = fit_angel_pca(contours,hierarchy_src,sr0)

for idx,angels in enumerate(angel_src):
    angel = angels - angel_temp[0]
    rotated_src = imutils.rotate(edges_src,angel)
    res = cv2.matchTemplate(rotated_src,edges_temlate, method)
    # xác dịnh tọa độ và vẽ khung cho template trên ảnh
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
    if(maxval>=0.88):
        mean_taget.append(center_cons[idx])
        angel_taget.append(angels)

    else:
        angel_inv = angels - angel_flip[0]
        rotated_src_inv = imutils.rotate(edges_src, angel_inv)
        res_inv = cv2.matchTemplate(rotated_src_inv, edges_template_flip, method)
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res_inv)
        if (maxval >= 0.88):
            print(maxval)
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
cv2.waitKey(0)
cv2. destroyAllWindows
