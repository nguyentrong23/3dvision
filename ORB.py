import cv2
import numpy as np

# Đọc ảnh và template
image = cv2.imread('path/to/image.jpg')
template = cv2.imread('path/to/template.jpg')

# Tọa độ tâm của object trong template và trên ảnh
template_center = (template.shape[1] // 2, template.shape[0] // 2)
object_centers = [(x, y) for x, y in zip([x_template, y_template], [x_image, y_image])]

# Kích thước của ROI (ví dụ: 50x50 pixel)
roi_size = (50, 50)

# Thực hiện template matching cho mỗi object
for center in object_centers:
    x, y = center
    roi = image[y - roi_size[1]//2: y + roi_size[1]//2, x - roi_size[0]//2: x + roi_size[0]//2]

    # Thực hiện template matching chỉ trong ROI
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

    # Lấy vị trí của điểm cực đại (vị trí của template trong ROI)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Chuyển vị trí từ tọa độ local của ROI về tọa độ global trên ảnh
    top_left = (max_loc[0] + x - roi_size[0]//2, max_loc[1] + y - roi_size[1]//2)
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    # Vẽ hình chữ nhật tại vị trí tìm được trên ảnh gốc
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Hiển thị ảnh kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
