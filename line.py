import cv2
import numpy as np
import sys

def crop_and_process_large_image(large_image_path, coordinates_str):
    large_image = cv2.imread(large_image_path)
    try:
        coordinates = list(map(int, coordinates_str.split(',')))
        x1, y1, x2, y2, x4, y4, x3, y3 = coordinates
        x = int(min(x1, x2, x3, x4))
        y = int(min(y1, y2, y3, y4))
        width = int(max(x1, x2, x3, x4) - x)
        height = int(max(y1, y2, y3, y4) - y)
        cropped_image = large_image[y:y + height, x:x + width]
        return cropped_image,large_image,x,y
    except:
        if not coordinates_str:
            cropped_image= large_image
            return  cropped_image,large_image,0,0
        print("coordinates erro")
        return 0,0,0,0


def preprocess_and_highlight_edges(image,sr00,thresh,x,y):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image,thresh, 255,  cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded_image, 50, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=15)
    mytring = ""
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(sr00, (x1+x, y1+y), (x2+x, y2+y), (0, 0, 255), 1, cv2.LINE_AA)
            note = str(x1+x) + " " + str(y1+y) + " " + str(x2+x) + " " + str(y2+y) + "/"
            mytring += note
    except:
        print("noline")
    return  mytring

# image_path = "datafornichi/src/back2.png"
# coordinates_str = ""
# sr0,sr,xtop_left,ytop_left = crop_and_process_large_image(image_path, coordinates_str)
# resutl=preprocess_and_highlight_edges(sr0,sr,90,xtop_left,ytop_left)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def main():
    if  len(sys.argv) < 4:
        print("missing path or thresh")
    elif len(sys.argv) == 4:
        path = sys.argv[1]
        try:
            thresh = int(sys.argv[2])
        except:
            thresh = 80
        if sys.argv[3]== "_":
            coordinates_str = ""
        else:
            coordinates_str = sys.argv[3]
        show = False
    elif len(sys.argv) <= 5:
        path = sys.argv[1]
        try:
            thresh = int(sys.argv[2])
        except:
            thresh = 80
        if sys.argv[3]== "_":
            coordinates_str = ""
        else:
            coordinates_str = sys.argv[3]
        show = True
    try:
        sr0,sr,xtop_left,ytop_left = crop_and_process_large_image(path, coordinates_str)
        resutl=preprocess_and_highlight_edges(sr0,sr,thresh,xtop_left,ytop_left)
        print(resutl)
        if show:
            cv2.imshow('ROI', sr0)
            cv2.imshow('SRC', sr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        return 'đường dẫn không chính xác'

if __name__ == "__main__":
    main()