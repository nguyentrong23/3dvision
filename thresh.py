import cv2
import numpy as np

def trackbar_callback(value, path):
    img = cv2.imread(path)
    thersh = value
    line,linedata= preprocess_and_highlight_edges(img,thersh)


def preprocess_and_highlight_edges(image,thresh):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,thresh, 255,  cv2.THRESH_BINARY)
    cv2.imshow('ThresholdedImage', thresholded_image)
    edges = cv2.Canny(thresholded_image, 50, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=15)
    lineData = {}
    try:
        for index,line in enumerate(lines) :
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            note = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "/"
            lineData[index] = note
    except:
        print("noline")
    return  image,lineData



path = "datafornichi/src/test_thick.png"
cv2.namedWindow('ThresholdedImage')
cv2.createTrackbar('Trackbar 1', 'ThresholdedImage', 0, 255, lambda x: trackbar_callback(x, path))
cv2.setTrackbarPos('Trackbar 1', 'ThresholdedImage', 50)

# Keep the window open
cv2.waitKey(0)
cv2.destroyAllWindows()
