import cv2
import numpy as np

# Global variables to store clicked points
point1 = (-1, -1)
point2 = (-1, -1)
lineData = {}  # Store line data globally

def trackbar_callback(value, path):
    img = cv2.imread(path)
    thresh = value
    line, _ = preprocess_and_highlight_edges(img, thresh)
    cv2.imshow('Thresholded Image', line)

def preprocess_and_highlight_edges(image,thresh):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,thresh, 255,  cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded_image, 50, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=15, maxLineGap=10)
    # lineData = {}
    try:
        for index,line in enumerate(lines) :
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            note = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
            lineData[index] = note
    except:
        print("noline")
    return  image,lineData

def distance_point_to_line(point, line):
    x1, y1, x2, y2 = line
    numerator = abs((y2 - y1) * point[0] - (x2 - x1) * point[1] + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = numerator / denominator if denominator != 0 else 0
    return distance

def mouse_callback(event, x, y, flags, param):
    global point1, point2
    if event == cv2.EVENT_LBUTTONDOWN:
        if point1 == (-1, -1):
            point1 = (x, y)
            closed_line=10
            closed_ind= 0
            for index, line in lineData.items():
                line_coordinates = [int(coord) for coord in line.split()]
                dist = distance_point_to_line((x, y), line_coordinates)
                if dist < closed_line:
                    closed_line = dist
                    closed_ind = index

            print(f"Clicked on line {closed_line}: {closed_ind} : {lineData[closed_ind]}")

        elif point2 == (-1, -1):
            point2 = (x, y)
            closed_line = 10
            closed_ind = 0
            for index, line in lineData.items():
                line_coordinates = [int(coord) for coord in line.split()]
                dist = distance_point_to_line((x, y), line_coordinates)
                if dist < closed_line:
                    closed_line = dist
                    closed_ind = index

            print(f"Clicked on line {closed_line}: {closed_ind} : {lineData[closed_ind]}")
            # Reset points for the next click
            point1 = (-1, -1)
            point2 = (-1, -1)

# Set the mouse callback function
path = "datafornichi/test.bmp"
cv2.namedWindow('Thresholded Image')  # Move this line above setMouseCallback
cv2.createTrackbar('Trackbar 1', 'Thresholded Image', 0, 255, lambda x: trackbar_callback(x, path))
cv2.setTrackbarPos('Trackbar 1', 'Thresholded Image', 50)

# Set the mouse callback function
cv2.setMouseCallback('Thresholded Image', mouse_callback)

# Keep the window open
cv2.waitKey(0)
cv2.destroyAllWindows()