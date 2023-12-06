import os
import cv2
import numpy as np
import math
import time

def get_gradient_sobel(image):
    img_src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(img_src, kernel, iterations=1)

    _, edges_src = cv2.threshold(dilated_image, 100, 140, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(edges_src, 80, 160)

    contours, hierarchy_src = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(edges_src, dtype=np.uint8)
    for cons in contours:
        area = cv2.contourArea(cons)
        if area > 1000:
            print(area)
            cv2.drawContours(mask, [cons], -1, (255), thickness=1)


    # sobel
    sobel_x = cv2.Sobel(edges_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edges_src, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))

    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[direc_angle == -90] = 255
    gradient_angle_flip[direc_angle == 90] = 255

    data_bottom = np.where(gradient_angle != 0)
    point_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(gradient_angle_flip != 0)
    point_top = np.column_stack((data_top[1], data_top[0]))

    lines_top = cv2.HoughLinesP(gradient_angle, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    lines_bot = cv2.HoughLinesP(gradient_angle_flip, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    try:
        for line in lines_top:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            print(x1, y1, x2, y2)
        for line in lines_bot:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
            print(x1, y1, x2, y2)
    except:
        print("noline")
    cv2.imshow('2',dilated_image)
    cv2.imshow('1', image)
    cv2.imshow('edges', edges)
    return edges,  point_top , point_bottom



def fit_pca(data, src):
    data = np.float32(data)
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
    cv2.circle(src, mean_point, 3, (0, 0, 255), -1)
    scale = 100
    vector1_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
    cv2.arrowedLine(src, mean_point, vector1_end, (0, 255,255), 1,cv2.LINE_AA)
    # Find min and max values along the direction of vector1_end
    min_val = np.min(np.dot(data - mean, eigenvectors.T))
    max_val = np.max(np.dot(data - mean, eigenvectors.T))
    # Draw a line covering the entire range of data along vector1_end
    line_start = (int(mean_point[0] + eigenvectors[0][0] * min_val), int(mean_point[1] + eigenvectors[0][1] * min_val))
    line_end = (int(mean_point[0] + eigenvectors[0][0] * max_val), int(mean_point[1] + eigenvectors[0][1] * max_val))
    cv2.line(src, line_start, line_end, (255,255, 0), 1,cv2.LINE_AA)
    cv2.imshow('3333333333333', src)
    return vector1_end,min_val,max_val


def find_longest_line(points):

    _, _, vx, vy, x0, y0, inliers = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # Tính điểm đầu và điểm cuối trên đường thẳng
    point1 = (int(x0 - 1000 * vx), int(y0 - 1000 * vy))
    point2 = (int(x0 + 1000 * vx), int(y0 + 1000 * vy))

    return point1, point2



# Đọc ảnh và tiền xử lý source
sr0 = cv2.imread("datafornichi/samp_lite.png")
edges, TopLine, Botline = get_gradient_sobel(sr0)

# # Gọi hàm con để tìm đường thẳng dài nhất
# line_start, line_end = find_longest_line(TopLine)
#
# # Vẽ đường thẳng lên ảnh để kiểm tra
# image = np.zeros((300, 2000, 3), dtype=np.uint8)
# cv2.line(image, line_start, line_end, (0, 255, 0), 2)
#
# # Hiển thị ảnh
# cv2.imshow('Longest Line', image)

# vector_top,xmin_top, xmax_top=fit_pca(TopLine,sr0)
# vector_bot,xmin_bot, xmax_bot=fit_pca( Botline,sr0)
# distance = cv2.norm(vector_top, vector_bot)
# print(f'xmin_top: {xmin_top}')
# print(f'xmax_top: {xmax_top}')
# print(f'xmin_bot: {xmin_bot}')
# print(f'xmax_bot {xmax_bot}')
# print(f'Khoảng cách giữa hai vector là: {distance}')
# print(f'độ phân giải ảnh là: {edges.shape[::]}')
cv2.waitKey(0)
cv2.destroyAllWindows()



