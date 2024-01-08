import cv2
import numpy as np

# Tạo ảnh đen (background)
sr = cv2.imread(r"datafornichi\template\h11_tem.bmp")

# Nhập tọa độ của hai điểm để vẽ đường thẳng đi qua
x1, y1 = 100, 50
x2, y2 = 400, 450

# Tính hệ số a và b của phương trình đường thẳng y = ax + b
a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

x_1 = 0
y_1 = int(a * x_1 + b)
y_2 = 0
x_2 = int(-b/a)
if y_1 >=0:
    start = (x_1,y_1)
    print(start)
elif x_1 >= 0:
    start = (x_2, y_2)
    print(start)

x_end1 = sr.shape[1] - 1
y_end1 = int(a * x_end1 + b)

y_end2 = sr.shape[0] - 1
x_end2 = int((y_end2-b)/a)

if y_end1<=sr.shape[0]:
    end = (x_end1,y_end1)
    print(end)
elif x_1 <=sr.shape[1]:
    end = (x_end2,y_end2)
    print(end)
cv2.line(sr, start, end, (0, 255, 255), 1, cv2.LINE_AA)
# Hiển thị ảnh với đường thẳng đã vẽ
cv2.imshow('Line on Image', sr)
cv2.waitKey(0)
cv2.destroyAllWindows()
