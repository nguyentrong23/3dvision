import cv2
import numpy as np

def trackbar_callback(value, path):
    threshold_value = 0
    beta = 0
    alpha = 0
    img = cv2.imread(path)

    trackbar_1 = cv2.getTrackbarPos('Trackbar 1', 'Thresholded Image')
    # trackbar_2 = cv2.getTrackbarPos('Trackbar 2', 'Thresholded Image')
    # trackbar_3 = cv2.getTrackbarPos('Trackbar 3', 'Thresholded Image')
    # Update global variables based on trackbar values
    threshold_value = trackbar_1
    # beta = trackbar_2 * 0.1  # Adjusting the step size to 0.1
    # alpha = trackbar_3 * 0.1  # Adjusting the step size to 0.1

    # Process image with updated parameters
    img_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(img_src, kernel, iterations=1)

    blurred = cv2.GaussianBlur(img_src, (5,5), 0)
    _, edges_src = cv2.threshold(dilated_image,threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(edges_src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    edges_src = cv2.pyrDown(edges_src)
    # cv2.imshow(f"detect {path}", edges_src)
    cv2.imshow('Thresholded Image', edges_src)

# Load the image
path = 'datafornichi/src1.jpeg'

# Create a window for the thresholded image
cv2.namedWindow('Thresholded Image')

# Create trackbars for threshold, contrast, and brightness
cv2.createTrackbar('Trackbar 1', 'Thresholded Image', 0, 255, lambda x: trackbar_callback(x, path))
cv2.createTrackbar('Trackbar 2', 'Thresholded Image', 1, 200, lambda x: trackbar_callback(x, path))
cv2.createTrackbar('Trackbar 3', 'Thresholded Image', 1, 50, lambda x: trackbar_callback(x, path))

# Initialize the trackbar positions
cv2.setTrackbarPos('Trackbar 1', 'Thresholded Image', 50)
cv2.setTrackbarPos('Trackbar 2', 'Thresholded Image', 10)
cv2.setTrackbarPos('Trackbar 3', 'Thresholded Image', 20)

# Keep the window open
cv2.waitKey(0)
cv2.destroyAllWindows()
