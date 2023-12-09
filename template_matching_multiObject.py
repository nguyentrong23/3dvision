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


def fit_angel_pca(contours, hierarchy, src):
    min = 1000
    output = {}
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        if hierarchy[0, index, 2] != -1:
            continue;
        area = cv2.contourArea(cnt)
        if area >= min:
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(src, mean_point, 5, (0, 0, 255), -1)
            output[tuple(mean.flatten())] = angel
            # output[angel] = mean
            scale = 100
            vector2_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            cv2.arrowedLine(src, mean_point, vector2_end, (0, 255, 0), 1,cv2.LINE_AA )
    return output, src

def tienxuly(sr0):
    img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _, edges_src = cv2.threshold(blurred, 60, 120, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return  contours,hierarchy,edges_src



# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("datafornichi/src.png")
contours, hierarchy_src,edges_src = tienxuly(sr0)


#  đọc  và tiền xử lý template
sr1= cv2.imread("datafornichi/template.png")
contours_temp, hierarchy_temp,edges_temp = tienxuly(sr1)

# init parameter
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = edges_temp.shape[::]
topleft = [0,0]
roi =max(h,w)
roi_size = (roi*2,roi*2)
angel_taget = []
mean_taget = []

# resolve angel problem by pca
template,template_show = fit_angel_pca(contours_temp,hierarchy_temp,sr1)
start_time = time.time()
object, src_show = fit_angel_pca(contours,hierarchy_src,sr0)

for angelt in template.values():
    for mean,angels in object.items():
        angel = angels - angelt
        #  cho roi từ mean rồi matching theo roi
        center_rotate = mean
        rotated_src = imutils.rotate(edges_src,angel,center_rotate)
        res = cv2.matchTemplate(rotated_src,edges_temp, method)
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
        print(maxval)
        print(angels)
        cv2.imshow(f"detect {mean}",  rotated_src)
        # loc = np.where(res >= 0.9)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(sr0, pt, (pt[0] + edges_temp.shape[1], pt[1] + edges_temp.shape[0]), (0, 255, 0), 2)

        if(maxval>=0.9):
            mean_taget.append(mean)
            angel_taget.append(angels)
            topleft = maxloc
            bottomright = (topleft[0] + w, topleft[1] + h)
            print(topleft)
            cv2.rectangle(sr0, topleft, bottomright, (0, 255, 255), 1)
        else:
            continue
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Thời gian chạy: {execution_time} giây")
#
# for index,point in enumerate(mean_taget):
#     mean_point = (int(round(point[0][0])), int(round(point[0][1])))
#     cv2.circle(sr0, mean_point, 5, (0, 255, 255), -1)
#     note = str(mean_point)
#     cv2.putText(sr0,note, mean_point, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
#     print(angel_taget[index])
cv2.imshow("detect sro",sr0)
cv2.imshow("edges_temlate",edges_temp)

cv2.waitKey(0)
cv2. destroyAllWindows

