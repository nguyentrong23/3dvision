import cv2
import numpy as np

def trackbar_callback(value, path):
    img = cv2.imread(path)
    img = remove_jig(img)
    thersh = value
    line,linedata= preprocess_and_highlight_edges(img,thersh)

def remove_jig(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 70, 70])
    upper_blue = np.array([130, 255, 255])
    # Tạo mask để chỉ giữ lại các pixel nằm trong khoảng giá trị màu xanh dương
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[(hsv[:, :, 0] >= lower_blue[0]) & (hsv[:, :, 0] <= upper_blue[0]) &
         (hsv[:, :, 1] >= lower_blue[1]) & (hsv[:, :, 1] <= upper_blue[1]) &
         (hsv[:, :, 2] >= lower_blue[2]) & (hsv[:, :, 2] <= upper_blue[2])] =255
    result = cv2.bitwise_or(img, mask)
    return result


def preprocess_and_highlight_edges(image,thresh):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,thresh, 255,  cv2.THRESH_BINARY)
    thresholded_image  = cv2.pyrDown( thresholded_image)
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



path = "datafornichi/src/back2.png"
cv2.namedWindow('ThresholdedImage')
cv2.createTrackbar('Trackbar 1', 'ThresholdedImage', 0, 255, lambda x: trackbar_callback(x, path))
cv2.setTrackbarPos('Trackbar 1', 'ThresholdedImage', 50)

# Keep the window open
cv2.waitKey(0)
cv2.destroyAllWindows()
