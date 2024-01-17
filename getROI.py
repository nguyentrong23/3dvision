import cv2
import matplotlib.pyplot as plt

# Hàm callback khi click chuột
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow('Select Points', img)

img = cv2.imread('datafornichi/protect/1.bmp')
# img= cv2.pyrDown(img)
cv2.namedWindow('Select Points')
points = []
cv2.setMouseCallback('Select Points', mouse_callback)
while True:
    cv2.imshow('Select Points', img)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == 4:
        break
cv2.destroyAllWindows()
print("Tọa độ của 4 điểm được chọn:")
for i, point in enumerate(points):
    print(f"Điểm {i+1}: {point}")
for point in points:
    cv2.circle(img, point, 4, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
