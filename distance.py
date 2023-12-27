import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ file
image = cv2.imread('datafornichi/src/realsense/h11.bmp', cv2.IMREAD_GRAYSCALE)

# Áp dụng bộ lọc Sobel theo chiều dọc
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Chuyển đổi giá trị âm thành dương
sobel_y = np.abs(sobel_y)
sobel_x = np.abs(sobel_x)

# Chuyển đổi kiểu dữ liệu về uint8 để hiển thị được hình ảnh
sobel_y = np.uint8(sobel_y)
sobel_x = np.uint8(sobel_x)

# Hiển thị hình ảnh gốc và kết quả
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(sobel_x, cmap='gray')
plt.title('x'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(sobel_y, cmap='gray')
plt.title('y'), plt.xticks([]), plt.yticks([])

plt.show()
