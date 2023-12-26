import cv2
import numpy as np
import math
import sys
import time


def distance_2_lines(coordinates, path):
    x1, y1, x2, y2, x4, y4, x3, y3 = coordinates
    point1_line2 = (x3, y3)
    point2_line2 = (x4, y4)
    midpoint_line1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    sr0 = cv2.imread(path)
    cv2.circle(sr0, (int(midpoint_line1[0]), int(midpoint_line1[1])), 3, (0, 0, 255), -1)

    # Phương trình đoạn thẳng đi qua point1_line2 và point2_line2
    m_line2, b_line2 = line_equation_through_points(point1_line2, point2_line2)
    m_perpendicular = -1 / m_line2
    b_perpendicular = midpoint_line1[1] - m_perpendicular * midpoint_line1[0]
    x_intersection = (b_line2 - b_perpendicular) / (m_perpendicular - m_line2)
    y_intersection = m_line2 * x_intersection + b_line2
    # Tính khoảng cách từ midpoint_line1 đến điểm giao
    distance = np.sqrt((midpoint_line1[0] - x_intersection) ** 2 + (midpoint_line1[1] - y_intersection) ** 2)
    # Draw line 2 on sr0
    cv2.line(sr0, (int(point1_line2[0]), int(point1_line2[1])), (int(point2_line2[0]), int(point2_line2[1])),
             (255, 0, 0), 1)

    # Draw the perpendicular line
    x_perpendicular = np.linspace(midpoint_line1[0], x_intersection, num=100)
    y_perpendicular = m_perpendicular * x_perpendicular + b_perpendicular
    for i in range(len(x_perpendicular) - 1):
        cv2.line(sr0, (int(x_perpendicular[i]), int(y_perpendicular[i])),(int(x_perpendicular[i + 1]), int(y_perpendicular[i + 1])), (0, 255, 0), 1)

    cv2.imshow("tttt", sr0)
    cv2.imwrite("tttt.png", sr0)
    return distance

def line_equation_through_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    else:
        return float('inf'), x1
path = "datafornichi/src/realsense/h11.bmp"
coordinates_str = "65,657,145,650,84,722,149,718"
coordinates = list(map(int, coordinates_str.split(',')))
distance = distance_2_lines(coordinates,path)
print(f"Khoảng cách là: {distance}")
cv2.waitKey(0)
cv2.destroyAllWindows()
# def main():
#     if  len(sys.argv) < 2:
#         print("missing coordinates")
#     elif len(sys.argv) == 2:
#         coordinates_str = sys.argv[1]
#         try:
#             coordinates = list(map(int, coordinates_str.split(',')))
#             distance = distance_2_lines(coordinates)
#             print(f"Khoảng cách là: {distance}")
#         except:
#             print("coordinates error")
#
# if __name__ == "__main__":
#     main()
