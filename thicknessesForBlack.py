import cv2
import numpy as np
import math

def get_gradient_sobel(image):
    img_src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    gray_image = cv2.morphologyEx(img_src, cv2.MORPH_CLOSE, kernel)
    # equalized_image = cv2.equalizeHist(img_src)
    blurred = cv2.GaussianBlur(gray_image, (3,3), 0)

    # thresh
    _, edges_src = cv2.threshold(blurred, 70, 140, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(edges_src, 100, 200)
    # sobel
    sobel_x = cv2.Sobel(edges_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edges_src, cv2.CV_64F, 0, 1, ksize=3)
    direc_angle = np.degrees(np.arctan2(sobel_y, sobel_x))

    gradient_angle = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle_flip = np.zeros_like(direc_angle, dtype=np.uint8)
    gradient_angle[direc_angle == -90] = 255
    gradient_angle_flip[direc_angle == 90] = 255

    data_bottom = np.where(gradient_angle != 0)
    data_bottom = np.column_stack((data_bottom[1], data_bottom[0]))
    data_top = np.where(gradient_angle_flip != 0)
    data_top = np.column_stack((data_top[1], data_top[0]))

    print(data_bottom)
    cv2.imshow('1',edges)
    cv2.imshow('2',gray_image)
    cv2.imshow('3', gradient_angle)
    cv2.imshow('4',gradient_angle_flip)
    return edges, data_top,data_bottom


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

# Đọc ảnh
# Nhận số nguyên từ người dùng
# print("dán đường dẫn ảnh: \n")
# path = str(input())
image = cv2.imread("datafornichi/pipe_lite.bmp")
# image = cv2.pyrUp(image)
edges, TopLine, Botline = get_gradient_sobel(image)
# vector_top,xmin_top, xmax_top=fit_pca(TopLine,image)
# vector_bot,xmin_bot, xmax_bot=fit_pca( Botline,image)
# distance = cv2.norm(vector_top, vector_bot)
# print(f'xmin_top: {xmin_top}')
# print(f'xmax_top: {xmax_top}')
# print(f'xmin_bot: {xmin_bot}')
# print(f'xmax_bot {xmax_bot}')
# print(f'Khoảng cách giữa hai vector là: {distance}')
# print(f'độ phân giải ảnh là: {edges.shape[::]}')
cv2.waitKey(0)
cv2.destroyAllWindows()


