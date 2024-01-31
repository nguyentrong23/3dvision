import ctypes
import cv2
import time
import numpy as np
import math
import imutils
import sys

class Point2d(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]
    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

class SingleTargetMatch(ctypes.Structure):
    _fields_ = [("ptLT", Point2d),
                ("ptRT", Point2d),
                ("ptRB", Point2d),
                ("ptLB", Point2d),
                ("ptCenter", Point2d),
                ("dMatchedAngle", ctypes.c_double),
                ("dMatchScore", ctypes.c_double)]
    def __str__(self) -> str:
        return f'SingleTargetMatch({self.ptLT}, {self.ptLB}, {self.ptRT}, {self.ptRB}, {self.ptCenter}, {self.dMatchedAngle}, {self.dMatchScore})'

# path = 'C:\\opencv\\build\\x64\\vc16\\bin\\ToolMatch.dll'
import os

if getattr(sys, 'frozen', False):
    # Chạy từ file exe đã được build
    path = os.path.join(os.path.dirname(sys.executable), 'ToolMatch.dll')
else:
    # Chạy từ môi trường phát triển
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_file_path, 'ToolMatch.dll')

dll_instance = None
dll_instance = ctypes.cdll.LoadLibrary(path)
match_cpp = dll_instance.match
match_cpp.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), 
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), 
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(SingleTargetMatch)), ctypes.POINTER(ctypes.c_int)]
match_cpp.restype = None
free_data_cpp = dll_instance.free_data
free_data_cpp.argtypes = [ctypes.POINTER(SingleTargetMatch)]
free_data_cpp.restype = None

def match(src, dst, target_number: int, max_overlap_ratio: float, score: float, tolerance_angle: float, min_reduce_area: int, tolerance1: float, tolerance2: float, tolerance3: float, tolerance4: float):
    global match_cpp
    src_data_ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    src_rows, src_cols, *_ = src.shape
    dst_data_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    dst_rows, dst_cols, *_ = dst.shape
    data_ptr = ctypes.POINTER(SingleTargetMatch)()
    length = ctypes.c_int()
    match_cpp(src_rows, src_cols, src_data_ptr, dst_rows, dst_cols, dst_data_ptr,
        target_number, max_overlap_ratio, score, tolerance_angle,
        min_reduce_area, tolerance1, tolerance2, tolerance3, tolerance4,
        ctypes.byref(data_ptr), ctypes.byref(length))

    # Chuyển con trỏ thành mảng các cấu trúc trong Python
    cast_obj = ctypes.cast(data_ptr, ctypes.POINTER(SingleTargetMatch * length.value))
    result_array = cast_obj.contents
    # free_data_cpp(data_ptr)
    return result_array


def   crop_temp(large_image_path, coordinates_str):
    large_image = cv2.imread(large_image_path)
    try:
        coordinates = list(map(int, coordinates_str.split(',')))
        x1, y1, x2, y2, x4, y4, x3, y3 = coordinates
        x = int(min(x1, x2, x3, x4))
        y = int(min(y1, y2, y3, y4))
        width = int(max(x1, x2, x3, x4) - x)
        height = int(max(y1, y2, y3, y4) - y)
        cropped_image = large_image[y:y + height, x:x + width]
        border_color = [255, 255, 255]
        angle_roi = np.arctan2(y2 - y1, x2 - x1)
        angle_roi = np.degrees(angle_roi)
        diagonal = math.sqrt(width ** 2 + height ** 2)
        paddwidth = int((diagonal - width)/2)
        paddheight = int((diagonal - height)/2)
        cropped_image = cv2.copyMakeBorder(cropped_image,paddheight,paddheight,paddwidth,paddwidth, cv2.BORDER_CONSTANT, value=border_color)
        x1 = x1 - x + paddwidth
        y1 = y1 - y + paddheight
        x2 = x2 - x + paddwidth
        y2 = y2 - y + paddheight
        x3 = x3 - x + paddwidth
        y3 = y3 - y + paddheight
        x4 = x4 - x + paddwidth
        y4 = y4 - y + paddheight
        point1 = rotate_point_in_image(cropped_image,(x1,y1), angle_roi)
        point2 = rotate_point_in_image(cropped_image,(x2,y2), angle_roi)
        point3 = rotate_point_in_image(cropped_image,(x3,y3), angle_roi)
        point4 = rotate_point_in_image(cropped_image,(x4,y4), angle_roi)
        rotated_src = imutils.rotate(cropped_image,angle_roi)
        # final crop
        xf = int(min(point1[0],point2[0], point3[0], point4[0]))
        yf = int(min(point1[1],point2[1], point3[1], point4[1]))
        widthf = int(max(point1[0],point2[0], point3[0], point4[0]) - xf)
        heightf = int(max(point1[1],point2[1], point3[1], point4[1]) - yf)
        cropped = rotated_src[yf:yf + heightf, xf:xf + widthf]
        # cv2.imshow('image_with_padding', rotated_src)
        # cv2.imshow('image',cropped)
        return cropped,angle_roi
    except:
        if not coordinates_str:
            cropped_image= large_image
            return  cropped_image,0
        print("coordinates erro")
        return 0,0

def rotate_point_in_image(image, point, angle_degrees):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 1]))
    new_x, new_y = rotated_point[:2].astype(int)
    newpoint = (int(new_x),int(new_y))
    return newpoint

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
        print("coordinates sr erro")
        return 0,0,0,0

def main():
    if  len(sys.argv) < 8:
        print("missing path:")
    elif len(sys.argv) == 8:
        path = sys.argv[1]
        coordinates_tem = sys.argv[2]
        if sys.argv[3] == "_":
            coordinates_src = ""
        else:
            coordinates_src = sys.argv[3]
        score = float(sys.argv[4])
        low_A = float(sys.argv[5])
        high_A = float(sys.argv[6])
        target = int(sys.argv[7])
        show = False

    elif len(sys.argv) >8:
        path = sys.argv[1]
        coordinates_tem = sys.argv[2]
        if sys.argv[3] == "_":
            coordinates_src = ""
        else:
            coordinates_src = sys.argv[3]
        score = float(sys.argv[4])
        low_A = float(sys.argv[5])
        high_A = float(sys.argv[6])
        target = int(sys.argv[7])
        show = True

    try:
        img, xs, ys = crop_and_process_large_image(path, coordinates_src)
        sr1, angel_start = crop_temp(path, coordinates_tem)

        src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(sr1, cv2.COLOR_BGR2GRAY)

        max_overlap_ratio = 0
        tolerance_angle = 180.0
        min_reduce_area = 256
        tolerance1 = 40
        tolerance2 = 60
        tolerance3 = -110
        tolerance4 = -100
        start_time = time.time()

        result = match(src, dst, target, max_overlap_ratio, score, tolerance_angle,
                       min_reduce_area, tolerance1, tolerance2, tolerance3, tolerance4)
        end_time = time.time()
        myst = ""
        for i, elem in enumerate(result):
            if low_A <= elem.dMatchedAngle <= high_A:
                # print(f'Angle : {elem.dMatchedAngle}')
                # print(f'Score : {elem.dMatchScore}')
                center = (int(elem.ptCenter.x), int(elem.ptCenter.y))
                # print("center", center)
                note = str(center[0]) + "," + str(center[1]) + "," + str(elem.dMatchedAngle) + "/"
                myst += note
                cv2.circle(img, (center), 3, (255, 123, 254), 3)
        print(f"Data:{myst}")
        processing_time = (end_time - start_time) * 1000
        processing_time = f"{processing_time:.2f}"
        print("Time: ", processing_time)
        if show:

                resized_img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow('V', resized_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("Code: 0")
    except Exception as e:
        print(f"Msg: {e}")
        print("Code: -1")



def deb():
    path = r"datafornichi/ttttttt/3.bmp"
    coordinates_src = ""
    coordinates_tem = "534,212,637,211,532,355,641,357"

    img, xs, ys = crop_and_process_large_image(path, coordinates_src)
    sr1, angel_start = crop_temp(path, coordinates_tem)

    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cvtColor(sr1, cv2.COLOR_BGR2GRAY)

    low_angle = -10
    high_angle = 10
    target_number = 100
    max_overlap_ratio = 0
    score = 0.95
    tolerance_angle = 180.0
    min_reduce_area = 256
    tolerance1 = 40
    tolerance2 = 60
    tolerance3 = -110
    tolerance4 = -100
    start_time = time.time()
    result = match(src, dst, target_number, max_overlap_ratio, score, tolerance_angle,
                   min_reduce_area, tolerance1, tolerance2, tolerance3, tolerance4)

    for i, elem in enumerate(result):
        if low_angle <= elem.dMatchedAngle <= high_angle:
            # cv2.putText(img,"X",( x4_c_x-50, x4_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
            # print(elem.ptLT.x, elem.ptLT.y)
            # print(elem.ptRT.x, elem.ptRT.y)
            # print(elem.ptRB.x, elem.ptRB.y)
            # print(elem.ptLB.x, elem.ptLB.y)
            # print(elem.ptCenter.x, elem.ptCenter.y)
            print(f'Angle : {elem.dMatchedAngle}')
            print(f'Score : {elem.dMatchScore}')

            x1 = (int(elem.ptLT.x), int(elem.ptLT.y))
            x2 = (int(elem.ptRT.x), int(elem.ptRT.y))
            x3 = (int(elem.ptRB.x), int(elem.ptRB.y))
            x4 = (int(elem.ptLB.x), int(elem.ptLB.y))
            center = (int(elem.ptCenter.x), int(elem.ptCenter.y))

            print("center",center)

            x1_c = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
            x2_c = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
            x3_c = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
            x4_c = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)

            # print(x1)
            # color_image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            # cv2.putText(img,"Angle:"+str(f"{elem.dMatchedAngle:.2f}"),(int(elem.ptCenter.x),int(elem.ptCenter.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),4)
            # cv2.line(img,(x1),(x2), (0, 255, 0), 3)
            # cv2.line(img,(x2),(x3), (0, 255, 0), 3)
            # cv2.line(img,(x3),(x4), (0, 255, 0), 3)
            # cv2.line(img,(x4),(x1), (0, 255, 0), 3)

            # cv2.circle(img,(x1_c), 3, (255, 0, 0), 3)
            # cv2.circle(img,(x2_c), 3, (0, 255, 0), 3)
            # cv2.circle(img,(x3_c), 3, (0, 0, 255), 3)
# mean+++++++++++++++++++++++++++++
            cv2.circle(img,(center), 3, (255, 123, 254), 3)
            # x4_c_x, x4_c_y = x4_c
            # x2_c_x, x2_c_y = x2_c
            # cv2.putText(img,"X",( x4_c_x-50, x4_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
            # cv2.putText(img,"y",( x2_c_x-50, x2_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
            # center_x, center_y = (center)
            # cv2.arrowedLine(img, (x1_c), (x2_c), (0, 255, 0), 1,cv2.LINE_AA)
            # cv2.arrowedLine(img, (x3_c), (x4_c), (0, 255, 0), 1,cv2.LINE_AA)
            # cv2.putText(img, str(i), (center_x - 50, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            # cv2.putText(img, str(i), (center_x - 50, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            # cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), 1)
            # cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), 1)
            # cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), 1)
            # cv2.circle(image_main,(e2), 4, (255, 0, 0), 2)
            resized_img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            processing_time = f"{processing_time:.2f}"
            print(processing_time)
            cv2.imshow('Video', resized_img)
            if cv2.waitKey(30) & 0xFF == 27:  # Nhấn phím Esc để thoát
                break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    deb()
        