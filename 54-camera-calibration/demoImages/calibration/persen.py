import cv2
import numpy as np

num_corners = (9,6)
square_size = 30
objp = np.zeros((np.prod(num_corners), 3), dtype=np.float32)
objp[:, :2] = np.indices(num_corners).T.reshape(-1, 2)
objp *= square_size
object_points = []
image_points = []
for i in range(1, 13):
    img = cv2.imread(f'chessboard{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, num_corners, None)
    if ret:
        object_points.append(objp)
        image_points.append(corners)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
img = cv2.imread('chessboard_front.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

if ret:
    pixel_mm_ratio = np.linalg.norm(corners[0] - corners[1]) / square_size
    print(f'Tỉ lệ pixel/mm: {pixel_mm_ratio}')
