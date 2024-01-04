import os
import cv2
import numpy as np
import math
import time
import sys
try:
    import imutils
except:
    os.system("pip install imutils")
    import imutils


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
        return cropped_image,x,y
    except:
        if not coordinates_str:
            cropped_image= large_image
            return  cropped_image,0,0
        print("coordinates erro")
        return 0,0,0,0


def fit_angel_pca(sr,thr):
    min = 3000
    output = {}
    # tien xu ly
    img_src = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3, 3), 0)
    _,src = cv2.threshold(blurred,thr, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for index, cnt in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue;
        # if hierarchy[0, index, 2] != -1:
        #     continue;
        area = cv2.contourArea(cnt)
        if area >= min:
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angel = math.atan2(eigenvectors[0][1], eigenvectors[0][0]) * (180 / math.pi)
            mean_point = (int(round(mean[0][0])), int(round(mean[0][1])))
            cv2.circle(sr, mean_point, 5, (0, 0, 255), -1)
            output[tuple(mean.flatten())] = angel
            scale = 100
            vector2_end = (int(mean_point[0] + eigenvectors[0][0] * scale), int(mean_point[1] + eigenvectors[0][1] * scale))
            cv2.line(sr, mean_point, vector2_end, (0, 255, 0), 1, cv2.LINE_AA)
            text = f"(angel: {angel})"
            cv2.putText(sr, text, (int(round(mean[0][0]) - 30), int(round(mean[0][1]) + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    # cv2.imshow('Original Image', sr)
    return output,src,contours,hierarchy

def remove_jig(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])
    # Tạo mask để chỉ giữ lại các pixel nằm trong khoảng giá trị màu xanh dương
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[(hsv[:, :, 0] >= lower_blue[0]) & (hsv[:, :, 0] <= upper_blue[0]) &
         (hsv[:, :, 1] >= lower_blue[1]) & (hsv[:, :, 1] <= upper_blue[1]) &
         (hsv[:, :, 2] >= lower_blue[2]) & (hsv[:, :, 2] <= upper_blue[2])] =255
    result = cv2.bitwise_or(img, mask)
    # cv2.imshow('Original Image', mask)
    # cv2.imshow('Image without Blue-Green Objects', result)
    return result

def padding(img,size):
    top, bottom, left, right = size, size,0,0
    border_color = [0, 0, 0]
    image_with_padding = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
    return image_with_padding

def rotate_point_in_image(image, point, angle_degrees):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 1]))
    new_x, new_y = rotated_point[:2].astype(int)
    newpoint = (int(new_x),int(new_y))
    return newpoint



def matching(edges_src,edges_tem,template,object,min_thresh,sr0):
    # # init parameter
    method = eval("cv2.TM_CCOEFF_NORMED")
    h, w = edges_tem.shape[:2]
    topleft = [0, 0]
    y_size = max(h, w)
    x_size = edges_src.shape[1]
    angel_target = []
    mean_target = []
    output = {}
    for angle_t in template.values():
        for index, (mean, angles) in enumerate(object.items()):
            # angle = angles - angle_t
            angle = angles
            rotated_src = imutils.rotate(edges_src,angle)
            roi_x = 0
            roi_y = int(mean[1] - y_size/2)
            roi_y = max(roi_y, 0)
            # Tạo ROI (Region of Interest)
            roi = rotated_src[roi_y:roi_y + y_size, roi_x:x_size]
            cv2.imshow('roi', roi)
            cv2.waitKey(0)

            if(roi_y+y_size) > edges_src.shape[0]:
                size = roi_y+y_size - edges_src.shape[0]
                roi =  padding(roi, size)
                roi_y = roi_y - size
            res = cv2.matchTemplate(roi, edges_tem, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(max_val)
            if max_val >= min_thresh:
                # sr0 = imutils.rotate(sr0, angle)
                topleft = max_loc
                topleft = (topleft[0], topleft[1] + roi_y)
                bottomright = (topleft[0] + w, topleft[1] + h)
                # cv2.rectangle(sr0, topleft, bottomright, (0, 255, 255), 1)
                center_x = (topleft[0] + bottomright[0]) // 2
                center_y = (topleft[1] + bottomright[1]) // 2
                # sr0 = imutils.rotate(sr0, -angle)

                m_target = (center_x, center_y)
                m_target = rotate_point_in_image(sr0, m_target, -angle)
                cv2.circle(sr0, (m_target[0], m_target[1]), 8, (0, 255, 255), -1)
                output[angles]= m_target
                # mean_target.append(m_target)
                # angel_target.append(angles)
    sr0 = cv2.pyrDown(sr0)
    sr0 = cv2.pyrDown(sr0)

    cv2.imshow('Original Image', sr0)
    # cv2.imwrite("kt_mean.png",sr0)
    return  output

path ="datafornichi/mid2.png"
coordinates_src = ""
coordinates_tem ="1332,866,1544,866,1544,989,1332,989"
thresh = 120


sr0,xs,ys = crop_and_process_large_image(path,coordinates_src)

sr1,xt,yt = crop_and_process_large_image(path,coordinates_tem)


sr1 = remove_jig(sr1)
sr0 = remove_jig(sr0)

template,edges_tem,contourt,_ = fit_angel_pca(sr1,thresh)
object,edges_src,contours,hierarchy = fit_angel_pca(sr0,thresh)
result = matching(edges_src,edges_tem,template,object,0.6,sr0)
myst=""
for k,v in result.items():
    note =str(v[0])+","+str(v[1])+","+str(k)+"/"
    myst+=note
print(myst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def main():
#     if  len(sys.argv) < 5:
#         print("missing path:")
#     elif len(sys.argv) == 5:
#         path = sys.argv[1]
#         coordinates_tem = sys.argv[2]
#         thresh = int(sys.argv[3])
#         min_val = float(sys.argv[4])
#         coordinates_src = ""
#     elif len(sys.argv) == 6:
#         path = sys.argv[1]
#         coordinates_tem = sys.argv[2]
#         thresh = int(sys.argv[3])
#         min_val = float(sys.argv[4])
#         coordinates_src = sys.argv[5]
#     try:
#         sr0,xs,ys = crop_and_process_large_image(path,coordinates_src)
#         sr1,xt,yt = crop_and_process_large_image(path,coordinates_tem)
#         sr1 = remove_jig(sr1)
#         sr0 = remove_jig(sr0)
#         template,edges_tem,contourt,_ = fit_angel_pca(sr1,thresh)
#         object,edges_src,contours,hierarchy = fit_angel_pca(sr0,thresh)
#
#         result = matching(edges_src, edges_tem, template, object, min_val, sr0)
#         myst = ""
#         for k, v in result.items():
#             note = str(v[0]) + "," + str(v[1]) + "," + str(k) + "/"
#             myst += note
#         print(myst)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#     except:
#         return 'đường dẫn không chính xác'
#
# if __name__ == "__main__":
#     main()