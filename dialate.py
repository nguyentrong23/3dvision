import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('datafornichi/pipe_lite.bmp')

blurred = cv2.pyrMeanShiftFiltering(image, 40,60)
kernel = np.ones((5,5), np.uint8)
dilated_image = cv2.dilate(blurred, kernel, iterations=1)
gray = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2GRAY)
# Áp dụng threshold để tạo ảnh nhị phân
_, thresh = cv2.threshold(gray, 100, 160, cv2.THRESH_BINARY_INV)

# Hiển thị ảnh gốc và ảnh sau khi dilation
cv2.imshow('blurred', blurred)
cv2.imshow('edges_src1',dilated_image)
cv2.imshow('edges_src2', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
