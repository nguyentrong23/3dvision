import cv2
import numpy as np
import imutils as imu

# doc anh va template
sr0 = cv2.imread("data/sample-2-3.bmp")
img_src = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)

sr1= cv2.imread("data/sample_for_template.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
best= 0
cv2.imshow("besT template angle",img_template)

# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]

threshold = 0.0
topleft = [0,0]
# resolve angel problem
for i in range(0,181,1):
    src_rotate = imu.rotate(img_src,i)
    res = cv2.matchTemplate(src_rotate,img_template, method)
    # xác dịnh tọa độ và vẽ khung cho template trên ảnh
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
    if(maxval>=threshold):
        print(threshold,':',i)
        best = i
        threshold = maxval
        topleft = maxloc

#     de ve hinh chu nhat thi can biet toa do 2 goc cheo
bottomright= (topleft[0]+w,topleft[1]+h)
sr0 = imu.rotate(img_src,best)
cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)

cv2.imshow("dectect",sr0)

cv2.waitKey(0)
cv2. destroyAllWindows