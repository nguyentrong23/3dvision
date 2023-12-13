import cv2
import numpy as np

def tienxuly(path):
    sr0 = cv2.imread(path)
    img_src = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    cv2.imshow(f"{path}", blurred)
    cv2.waitKey(0)
    return blurred


def ORB(img1, img2, threshold):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    img1 = cv2.drawKeypoints(img1, des1, None)
    img2 = cv2.drawKeypoints(img2, des2, None)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    print(matches)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [match for match in matches if match.distance < threshold]
    if good_matches:
        matching_points = [kp1[match.queryIdx].pt for match in good_matches]
        matching_points = np.array(matching_points, dtype=np.float32)
        x, y, w, h = cv2.boundingRect(matching_points)
        # Vẽ bounding box lên hình 1
        result_img = img1.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        result_img = cv2.pyrDown(result_img)
        result_img = cv2.pyrDown(result_img)
        cv2.imshow("Matching Region", result_img)
    else:
        print("No good matches found.")


# Đọc ảnh và tiền xử lý source
path_src = "datafornichi/src/t2.png"
edges_src = tienxuly(path_src)

#  đọc  và tiền xử lý template
path_tem = "datafornichi/template/tem.png"
edges_temp = tienxuly(path_tem)

ORB(edges_src,edges_temp,30)


cv2.waitKey(0)
cv2.destroyAllWindows()
