import cv2
import numpy as np

def preprocess_and_highlight_edges(image_path):
    image = cv2.imread(image_path)
    blurred = cv2.pyrMeanShiftFiltering(image, 20, 30)
    blurred_image = cv2.GaussianBlur(blurred, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 80, 255,  cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresholded_image, 50, 150)
    cv2.imshow('edges',edges)

    data = np.where(edges != 0)
    data = np.column_stack((data[1], data[0]))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=15)
    mytring = ""
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            note = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "/"
            mytring += note

    except:
        print("noline")
    cv2.imshow('1', image)
    print(mytring)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data, mytring

# Gọi hàm với đường dẫn của ảnh
image_path = "datafornichi/samp_lite.png"
preprocess_and_highlight_edges(image_path)
