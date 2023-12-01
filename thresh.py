import cv2
import numpy as np

def find_longest_line(points):

    _, _, vx, vy, x0, y0, inliers = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # Tính điểm đầu và điểm cuối trên đường thẳng
    point1 = (int(x0 - 1000 * vx), int(y0 - 1000 * vy))
    point2 = (int(x0 + 1000 * vx), int(y0 + 1000 * vy))

    return point1, point2

def main():
    # Tạo danh sách tọa độ điểm
    points = np.array([[531, 172], [527, 173], [528, 173], [1725, 269], [1726, 269], [1727, 269]])

    # Gọi hàm con để tìm đường thẳng dài nhất
    line_start, line_end = find_longest_line(points)

    # Vẽ đường thẳng lên ảnh để kiểm tra
    image = np.zeros((300, 2000, 3), dtype=np.uint8)
    cv2.line(image, line_start, line_end, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow('Longest Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()