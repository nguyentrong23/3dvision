# -*- coding: utf-8 -*-

import os
import ctypes
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import numpy as np
from PyQt5.QtCore import QByteArray
import time
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
# from numba import jit
# from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
# from PyQt5.QtGui import QImage, QPixmap, QPainter
# from PyQt5.QtCore import Qt
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
# Lấy đường dẫn của tệp .py đang thực thi
current_file_path = os.path.dirname(os.path.abspath(__file__))
path = f'{current_file_path}\\ToolMatch.dll'

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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1099, 730)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 251, 671))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet("background-color: rgb(0, 85, 255);\n"
"color: rgb(255, 0, 0);")
        self.groupBox.setObjectName("groupBox")
        self.La_Srceen1 = QtWidgets.QLabel(self.groupBox)
        self.La_Srceen1.setGeometry(QtCore.QRect(10, 30, 231, 191))
        self.La_Srceen1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.La_Srceen1.setText("")
        self.La_Srceen1.setObjectName("La_Srceen1")
        self.Bnt_Open_Template = QtWidgets.QPushButton(self.groupBox)
        self.Bnt_Open_Template.setGeometry(QtCore.QRect(10, 230, 111, 24))
        self.Bnt_Open_Template.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Bnt_Open_Template.setObjectName("Bnt_Open_Template")
        self.Bnt_Open_Image = QtWidgets.QPushButton(self.groupBox)
        self.Bnt_Open_Image.setGeometry(QtCore.QRect(130, 230, 111, 24))
        self.Bnt_Open_Image.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Bnt_Open_Image.setObjectName("Bnt_Open_Image")
        self.Bnt_Trigger = QtWidgets.QPushButton(self.groupBox)
        self.Bnt_Trigger.setGeometry(QtCore.QRect(10, 300, 111, 24))
        self.Bnt_Trigger.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Bnt_Trigger.setObjectName("Bnt_Trigger")
        self.Bnt_Live = QtWidgets.QPushButton(self.groupBox)
        self.Bnt_Live.setGeometry(QtCore.QRect(130, 300, 111, 24))
        self.Bnt_Live.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Bnt_Live.setObjectName("Bnt_Live")
        self.Le_Exposuretime = QtWidgets.QLineEdit(self.groupBox)
        self.Le_Exposuretime.setGeometry(QtCore.QRect(130, 330, 111, 22))
        self.Le_Exposuretime.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Le_Exposuretime.setObjectName("Le_Exposuretime")
        self.Le_Timeout = QtWidgets.QLineEdit(self.groupBox)
        self.Le_Timeout.setGeometry(QtCore.QRect(130, 360, 111, 22))
        self.Le_Timeout.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Le_Timeout.setObjectName("Le_Timeout")
        self.La_Gain = QtWidgets.QLineEdit(self.groupBox)
        self.La_Gain.setGeometry(QtCore.QRect(130, 390, 111, 22))
        self.La_Gain.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.La_Gain.setObjectName("La_Gain")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 330, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 360, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 390, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_4.setObjectName("label_4")
        self.La_NumbertoFind = QtWidgets.QLineEdit(self.groupBox)
        self.La_NumbertoFind.setGeometry(QtCore.QRect(130, 420, 111, 22))
        self.La_NumbertoFind.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.La_NumbertoFind.setObjectName("La_NumbertoFind")
        self.La_Threshold = QtWidgets.QLineEdit(self.groupBox)
        self.La_Threshold.setGeometry(QtCore.QRect(130, 450, 111, 22))
        self.La_Threshold.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.La_Threshold.setObjectName("La_Threshold")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(10, 420, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 450, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_6.setObjectName("label_6")
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox)
        self.horizontalSlider.setGeometry(QtCore.QRect(60, 510, 161, 16))
        self.horizontalSlider.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.horizontalSlider.setMaximum(180)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.La_Anglemibus = QtWidgets.QLabel(self.groupBox)
        self.La_Anglemibus.setGeometry(QtCore.QRect(220, 540, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.La_Anglemibus.setFont(font)
        self.La_Anglemibus.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.La_Anglemibus.setMidLineWidth(5)
        self.La_Anglemibus.setAlignment(QtCore.Qt.AlignCenter)
        self.La_Anglemibus.setObjectName("La_Anglemibus")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.groupBox)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(60, 540, 161, 16))
        self.horizontalSlider_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.horizontalSlider_3.setMinimum(0)
        self.horizontalSlider_3.setMaximum(180)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.La_Angleplus = QtWidgets.QLabel(self.groupBox)
        self.La_Angleplus.setGeometry(QtCore.QRect(220, 510, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.La_Angleplus.setFont(font)
        self.La_Angleplus.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.La_Angleplus.setMidLineWidth(5)
        self.La_Angleplus.setAlignment(QtCore.Qt.AlignCenter)
        self.La_Angleplus.setObjectName("La_Angleplus")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 510, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_7.setObjectName("label_7")
        self.label_25 = QtWidgets.QLabel(self.groupBox)
        self.label_25.setGeometry(QtCore.QRect(10, 540, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_25.setFont(font)
        self.label_25.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_25.setObjectName("label_25")
        self.La_Step_Angle = QtWidgets.QLineEdit(self.groupBox)
        self.La_Step_Angle.setGeometry(QtCore.QRect(130, 480, 111, 22))
        self.La_Step_Angle.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.La_Step_Angle.setObjectName("La_Step_Angle")
        self.label_26 = QtWidgets.QLabel(self.groupBox)
        self.label_26.setGeometry(QtCore.QRect(10, 480, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_26.setFont(font)
        self.label_26.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.groupBox)
        self.label_27.setGeometry(QtCore.QRect(10, 630, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_27.setFont(font)
        self.label_27.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_27.setObjectName("label_27")
        self.La_Circle_Time = QtWidgets.QLineEdit(self.groupBox)
        self.La_Circle_Time.setGeometry(QtCore.QRect(100, 630, 141, 22))
        self.La_Circle_Time.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.La_Circle_Time.setObjectName("La_Circle_Time")
        self.label_28 = QtWidgets.QLabel(self.groupBox)
        self.label_28.setGeometry(QtCore.QRect(10, 600, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_28.setFont(font)
        self.label_28.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_28.setObjectName("label_28")
        self.Le_Total_Object = QtWidgets.QLineEdit(self.groupBox)
        self.Le_Total_Object.setGeometry(QtCore.QRect(100, 600, 141, 22))
        self.Le_Total_Object.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.Le_Total_Object.setObjectName("Le_Total_Object")
        self.La_Srceen2_2 = QtWidgets.QLabel(self.groupBox)
        self.La_Srceen2_2.setGeometry(QtCore.QRect(270, 100, 821, 671))
        self.La_Srceen2_2.setStyleSheet("background-color: rgb(0, 85, 255);")
        self.La_Srceen2_2.setText("")
        self.La_Srceen2_2.setObjectName("La_Srceen2_2")
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 570, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.label_15.setObjectName("label_15")
        self.horizontalSlider_6 = QtWidgets.QSlider(self.groupBox)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(50, 570, 171, 16))
        self.horizontalSlider_6.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.horizontalSlider_6.setMinimum(-180)
        self.horizontalSlider_6.setMaximum(180)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.La_Angleplus_2 = QtWidgets.QLabel(self.groupBox)
        self.La_Angleplus_2.setGeometry(QtCore.QRect(220, 570, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.La_Angleplus_2.setFont(font)
        self.La_Angleplus_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.La_Angleplus_2.setMidLineWidth(5)
        self.La_Angleplus_2.setAlignment(QtCore.Qt.AlignCenter)
        self.La_Angleplus_2.setObjectName("La_Angleplus_2")
        self.Bnt_Train_Template = QtWidgets.QPushButton(self.groupBox)
        self.Bnt_Train_Template.setGeometry(QtCore.QRect(10, 260, 111, 24))
        self.Bnt_Train_Template.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.Bnt_Train_Template.setObjectName("Bnt_Train_Template")
        self.La_Srceen2 = QtWidgets.QLabel(self.centralwidget)
        self.La_Srceen2.setGeometry(QtCore.QRect(270, 10, 821, 671))
        self.La_Srceen2.setStyleSheet("background-color: rgb(0, 85, 255);")
        self.La_Srceen2.setText("")
        self.La_Srceen2.setObjectName("La_Srceen2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1099, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Bnt_Train_Template.clicked.connect(self.load_image)
        self.Bnt_Open_Image.clicked.connect(self.load_image2)
        self.Bnt_Trigger.clicked.connect(self.Trigger)
        self.Bnt_Open_Template.clicked.connect(self.Creat_ROI)

        self.horizontalSlider_6.valueChanged.connect(self.update_slider_value)
        self.horizontalSlider.valueChanged.connect(self.update_slider_value2)
        self.horizontalSlider_3.valueChanged.connect(self.update_slider_value3)
        self.horizontalSlider.setValue(15)
        self.horizontalSlider_3.setValue(15)
        self.La_Threshold.setText("0.5")
        self.La_NumbertoFind.setText("1")
        self.Object_count=0
        self.dragging_x1 = False
        self.start_drag_pos = None
        #self.x1 = (80, 80) 
         # Tọa độ của x1
        self.static_roi = False
        self.static_roi2 = True
        self.communication_flag = False
        self.follow_lock = threading.Lock()
        self.thread = threading.Thread(target=self.Reload_ROI)
        self.thread.daemon = True
        self.thread.start()
         # Thêm QLabel mới để hiển thị tọa độ chuột
        #self.La_Srceen2 = QLabel(self.groupBox)
        #self.La_Srceen2.setGeometry(QtCore.QRect(300, 440, 231, 30))
        # self.La_Srceen2.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.La_Srceen2.setAlignment(QtCore.Qt.AlignCenter)
        #self.La_Srceen2.setObjectName("La_MouseCoordinates")

        # self.La_Srceen2 = QtWidgets.QLabel(self.centralwidget)
        # self.La_Srceen2.setGeometry(QtCore.QRect(270, 10, 821, 671))
        # self.La_Srceen2.setStyleSheet("background-color: rgb(0, 85, 255);")
        # self.La_Srceen2.setText("")
        # self.La_Srceen2.setObjectName("La_Srceen2")

        # Kết nối sự kiện chuột vào phương thức xử lý
        #self.La_Srceen2.mousePressEvent = self.mouse_move_event
        self.La_Srceen2.mouseMoveEvent = self.mouse_move_event
        self.La_Srceen2.mouseReleaseEvent = self.mouse_release_event
        self.e =0
        self.r =True
        self.img3 =None
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Template_Matching"))
        self.Bnt_Open_Template.setText(_translate("MainWindow", "Creat_ROI"))
        self.Bnt_Open_Image.setText(_translate("MainWindow", "Open_Image"))
        self.Bnt_Trigger.setText(_translate("MainWindow", "Trigger"))
        self.Bnt_Live.setText(_translate("MainWindow", "Live"))
        self.label_2.setText(_translate("MainWindow", "Exposure_Time:"))
        self.label_3.setText(_translate("MainWindow", "Time Out:"))
        self.label_4.setText(_translate("MainWindow", "Gain:"))
        self.label_5.setText(_translate("MainWindow", "Number To Find:"))
        self.label_6.setText(_translate("MainWindow", "Threshold:"))
        self.La_Anglemibus.setText(_translate("MainWindow", "0"))
        self.La_Angleplus.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "Angle+"))
        self.label_25.setText(_translate("MainWindow", "Angle-"))
        self.label_26.setText(_translate("MainWindow", "Step_Angle"))
        self.label_27.setText(_translate("MainWindow", "Circle Time:"))
        self.label_28.setText(_translate("MainWindow", "Total_Object:"))
        self.label_15.setText(_translate("MainWindow", "Angle"))
        self.La_Angleplus_2.setText(_translate("MainWindow", "0"))
        self.Bnt_Train_Template.setText(_translate("MainWindow", "Train_Template"))
    def Trigger(self):
        start_time = time.time()
        img = self.img3# template image
        image_main = self.image_main.copy()

        if img is not None:

            Angle_To_Find = self.value2# Giá trị ban đầu
            text = self.La_Threshold.text()
            Number_To_Find=self.La_NumbertoFind.text()
            Number_To_Find= int(Number_To_Find)
            threshold= float(text)
            

            dst= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_main2 = image_main.copy()
            src =cv2.cvtColor(image_main2, cv2.COLOR_BGR2GRAY)
            print("agtf",Angle_To_Find)
            target_number =Number_To_Find
            max_overlap_ratio = 0
            score = threshold
            tolerance_angle = Angle_To_Find
            min_reduce_area = 256
            tolerance1 = 40
            tolerance2 = 60
            tolerance3 = -110
            tolerance4 = -100
            result = match(src, dst, target_number, max_overlap_ratio, score, tolerance_angle,
                min_reduce_area, tolerance1, tolerance2, tolerance3, tolerance4)     
            #print(result)
            if result:
                #print("Result:")
                i = 0
                for elem in result:
                    i +=1
                    # cv2.putText(img,"X",( x4_c_x-50, x4_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
                    # print(elem.ptLT.x, elem.ptLT.y)
                    # print(elem.ptRT.x, elem.ptRT.y)
                    # print(elem.ptRB.x, elem.ptRB.y)
                    # print(elem.ptLB.x, elem.ptLB.y)
                    # print(elem.ptCenter.x, elem.ptCenter.y)
                    # print(f'Angle : {elem.dMatchedAngle}')
                    # print(f'Score : {elem.dMatchScore}')

                    x1= (int(elem.ptLT.x),int(elem.ptLT.y))
                    x2= (int(elem.ptRT.x),int(elem.ptRT.y))
                    x3= (int(elem.ptRB.x),int(elem.ptRB.y))
                    x4= (int(elem.ptLB.x),int(elem.ptLB.y))
                    center =(int(elem.ptCenter.x),int(elem.ptCenter.y))

                    x1_c =  ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
                    x2_c =  ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
                    x3_c =  ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
                    x4_c =  ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)
                    
                   
                    #print(x1)
                    #color_image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
                    #cv2.putText(img,"Angle:"+str(f"{elem.dMatchedAngle:.2f}"),(int(elem.ptCenter.x),int(elem.ptCenter.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),4)
                    # cv2.line(img,(x1),(x2), (0, 255, 0), 3)
                    # cv2.line(img,(x2),(x3), (0, 255, 0), 3)
                    # cv2.line(img,(x3),(x4), (0, 255, 0), 3)
                    # cv2.line(img,(x4),(x1), (0, 255, 0), 3)

                    # cv2.circle(img,(x1_c), 3, (255, 0, 0), 3)
                    # cv2.circle(img,(x2_c), 3, (0, 255, 0), 3)
                    # cv2.circle(img,(x3_c), 3, (0, 0, 255), 3)
                    # cv2.circle(img,(center), 3, (255, 123, 254), 3)
                    x4_c_x,x4_c_y =x4_c
                    x2_c_x,x2_c_y =x2_c
                    # cv2.putText(img,"X",( x4_c_x-50, x4_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
                    # cv2.putText(img,"y",( x2_c_x-50, x2_c_y+8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),8)
                    

                    center_x,center_y = (center)
                    cv2.arrowedLine(image_main,(x1_c),(x2_c), (0, 255, 0), 4)
                    cv2.arrowedLine(image_main,(x3_c),(x4_c), (0, 255, 0), 5)
                    cv2.putText(image_main, str(i), (center_x -50,center_y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
                    # cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), 1)
                    # cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), 1)
                    # cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), 1)
                    # cv2.circle(image_main,(e2), 4, (255, 0, 0), 2)
        resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
        img_height, img_width, img_channel = resized_img3.shape
        q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
        pixmap2 = QPixmap.fromImage(q_image2)
        self.La_Srceen2.setPixmap(pixmap2)
         
        #resized_img2= cv2.resize(rotated_image, (231, 231), interpolation=cv2.INTER_AREA)  # Không thay đổi kích thước của ảnh sau khi xoay
        # img_height, img_width, img_channel = resized_img2.shape
        # q_image = QImage(resized_img2.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
        # pixmap = QPixmap.fromImage(q_image)
        # self.La_Srceen1.setPixmap(pixmap)
        self.Le_Total_Object.setText(str(self.Object_count))
        #end_value=last_value
        end_time = time.time()
        processing_time = (end_time - start_time)*1000
        #processing_time = f"{processing_time:.2f}"
        processing_time = "{:.2f}".format(processing_time)
        #print(processing_time)   
        self.La_Circle_Time.setText(processing_time)
        self.Le_Total_Object.setText(str(self.Object_count))
    def mouse_release_event(self, event):
         self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
         # image_main = self.image_main.copy()
         # x1=np.array(self.x5)
         # x2=np.array(self.x6)
         # x3=np.array(self.x7)
         # x4=np.array(self.x8)
         # cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0), 1)
         # cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), 1)
         # cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), 1)
         # cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), 1) 
         # resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
         # img_height, img_width, img_channel = resized_img3.shape
         # q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
         # pixmap2 = QPixmap.fromImage(q_image2)
         # self.La_Srceen2.setPixmap(pixmap2) 
        #self.upload_ROI()
        #print("oioi")
       
    def mouse_move_event(self, event):
       self.x_mouse=event.x()
       self.y_mouse=event.y()
       self.calib_width = self.width_camera/821
       self.calib_height = self.height_camera/671
       #print(self.height_camera) 
       #self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))#dau mui toa do       
       image_main = self.image_main.copy()
       if self.resolution_camera >1:
           self.x= self.x_mouse*self.calib_width
           self.y= self.y_mouse*self.calib_height
           x=self.x
           y=self.y
           #print(x,y) 
       elif self.resolution_camera ==1: 
          
           self.x= self.x_mouse//1.281
           self.y= self.y_mouse//1.4
           x=self.x
           y=self.y 

       # e1 = self.e1
       # e2 = self.e2
       # e3 = self.e3
    #    self.width=100
    #    self.height=80
    #    self.e =e2
    #    center_x = int(x)
    #    center_y = int(y)       #print(e2)
    #    x1 = self.new_center_x
    #    y1 = self.new_center_y  # origin 
    #    x2, y2 = self.e2
    #    x3= self.x
    #    y3 = self.y
    # # Tính góc giữa điểm đầu và điểm cuối theo chiều kim đồng hồ
    #    angle = np.arctan2(y3 - y1, x3 - x1) - np.arctan2(y2 - y1, x2 - x1)
    # Đảm bảo góc nằm trong khoảng từ 0 đến 360 độ
       # if angle < 0:
       #   angle += 2 * np.pi
       # angle = angle * 180 / np.pi
       # print(angle)
       # print(x)
       #print(self.top_left,self.bottom_left,self.top_right,self.bottom_right)
       if self.top_left <= self.x <= self.top_right and self.bottom_left <= self.y<= self.bottom_right:#hàm di chuyển  TAM ROI
             self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))#dau mui toa do
             center_x = int(self.x)
             center_y = int(self.y)
             #print("center")
             #angle2= self.angle2

             x=self.x
             y=self.y
             x1=self.x1
             x2=self.x2
             x3=self.x3
             x4=self.x4
                # Tính vector của đoạn thẳng ab


            
             e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
             e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
             e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
             e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)


             self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
             self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
             self.new_center_x= int(x2[0]+self.new_width//2)
             self.new_center_y =int(x2[1]+self.new_height//2)
             x2[0]=x2[0]-(self.new_center_x-self.x)
             x1[0]=x1[0]-(self.new_center_x-self.x)
             x3[0]=x3[0]-(self.new_center_x-self.x)
             x4[0]=x4[0]-(self.new_center_x-self.x)

             x2[1]=x2[1]-(self.new_center_y-self.y)
             x1[1]=x1[1]-(self.new_center_y-self.y)
             x3[1]=x3[1]-(self.new_center_y-self.y)
             x4[1]=x4[1]-(self.new_center_y-self.y)



             self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
             self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
             self.new_center_x= int(x2[0]+self.new_width//2)
             self.new_center_y =int(x2[1]+self.new_height//2)
             e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
             e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
             e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
             e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)

             self.top_left_x = e1[0]-self.distance_click1
             self.top_left_y = e1[1]-self.distance_click1
             self.bottom_right_x =e1[0]+self.distance_click1
             self.bottom_right_Y =e1[1]+self.distance_click1


             self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
             self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
             self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
             self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
       
             self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
             self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
             self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
             self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

             self.top_left4 = e2[0]-self.distance_click2
             self.bottom_left4 =e2[1] -self.distance_click2
             self.top_right4 = e2[0]+self.distance_click2
             self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

             self.top_left = center_x-self.distance_click2#center_x-2#x1
             self.bottom_left =center_y -self.distance_click2#center_y-2#y1
             self.top_right = center_x+self.distance_click2#center_x+2#x2
             self.bottom_right = center_y +self.distance_click2#center_y+2#y2


             self.top_left5 = e4[0]-self.distance_click2
             self.bottom_left5 =e4[1] -self.distance_click2
             self.top_right5 = e4[0]+self.distance_click2
             self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 
             cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0),self.thickness)
             cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), self.thickness)
             cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), self.thickness)
             cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), self.thickness) 

             # cv2.putText(image_main ,"X2: ",(x2[0]-4,x2[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
             # cv2.putText(image_main ,"X1: ",(x1[0]-4,x1[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
             # cv2.putText(image_main ,"X3: ",(x3[0]-4,x3[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
             # cv2.putText(image_main ,"X4: ",(x4[0]-4,x4[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)

             cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

             cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
    #
             cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
             cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)
             #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)
             self.g =e2
             cv2.circle(image_main,(self.new_center_x,self.new_center_y), 1, (255, 0, 0), -1)
            
             resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
             img_height, img_width, img_channel = resized_img3.shape
             q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
             pixmap2 = QPixmap.fromImage(q_image2)
             self.La_Srceen2.setPixmap(pixmap2) 

       elif self.top_left2 <= self.x <= self.top_right2 and self.bottom_left2 <= self.y<= self.bottom_right2:
          # self.x_mouse=event.x()
          # self.y_mouse=event.y()
          #print("doc")
          # self.x= self.x_mouse//1.281
          # self.y= self.y_mouse//1.4
       	  self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.SizeVerCursor))#dau mui ten doc

          x1=self.x1
          x2=self.x2
          x3=self.x3
          x4=self.x4
          #print(type(self.x1))
          #x1,x2,x3,x4 =a[0]
          x2[1]=x2[1]-(x2[1]-self.y)
          x3[1]=x3[1]-(x3[1]-self.y)



          e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
          e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
          e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
          e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)
                  
          # cv2.putText(image_main ,"X2: ",(x2[0]-4,x2[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X3: ",(x3[0]-4,x3[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X1: ",(x1[0]-4,x1[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X4: ",(x4[0]-4,x4[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)


          self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
          self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
          self.new_center_x= int(x2[0]+self.new_width//2)
          self.new_center_y =int(x2[1]+self.new_height//2)

          cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0),self.thickness)
          cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), self.thickness) 

          cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

          cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
#
          cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)

          #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)
          cv2.circle(image_main,(self.new_center_x,self.new_center_y), 1, (255, 0, 0), -1)

          # self.top_left_x = e1[0]-self.distance_click1
          # self.top_left_y = e1[1]-self.distance_click1
          # self.bottom_right_x =e1[0]+self.distance_click1
          # self.bottom_right_Y =e1[1]+self.distance_click1


          self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
          self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
          self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
          self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
   
          self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
          self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
          self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
          self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

          self.top_left4 = e2[0]-self.distance_click2
          self.bottom_left4 =e2[1] -self.distance_click2
          self.top_right4 = e2[0]+self.distance_click2
          self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

          self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
          self.bottom_left =self.new_center_y -self.distance_click2#center_y-2#y1
          self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
          self.bottom_right = self.new_center_y +self.distance_click2#center_y+2#y2
          #self.new_center_x=center_x
          self.top_left5 = e4[0]-self.distance_click2
          self.bottom_left5 =e4[1] -self.distance_click2
          self.top_right5 = e4[0]+self.distance_click2
          self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 

          resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
          img_height, img_width, img_channel = resized_img3.shape
          q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
          pixmap2 = QPixmap.fromImage(q_image2)
          self.La_Srceen2.setPixmap(pixmap2) 
         #self.La_Srceen2.mouseReleaseEvent = self.mouse_release_event 

       elif self.top_left3 <= self.x <= self.top_right3 and self.bottom_left3 <= self.y<= self.bottom_right3:
          self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))#dau mui ten ngang 
        #  angle2 =0
          x1=self.x1
          x2=self.x2
          x3=self.x3
          x4=self.x4
          x2[0]=x2[0]-(x2[0]-self.x)
          x1[0]=x1[0]-(x1[0]-self.x)
          #print("ngang1")

          # self.top_left = center_x-10#center_x-2#x1
          # self.bottom_left =center_y -10#center_y-2#y1
          # self.top_right = center_x+10#center_x+2#x2
          # self.bottom_right = center_y +10#center_y+2#y2
          e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
          e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
          e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
          e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)


         # cv2.putText(image_main ,"X2: ",(x2[0]-4,x2[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
         # cv2.putText(image_main ,"X1: ",(x1[0]-4,x1[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
         # cv2.putText(image_main ,"X3: ",(x3[0]-4,x3[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
         # cv2.putText(image_main ,"X4: ",(x4[0]-4,x4[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)



          self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
          self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
          self.new_center_x= int(x2[0]+self.new_width//2)
          self.new_center_y =int(x2[1]+self.new_height//2)

          cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0),self.thickness)
          cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), self.thickness) 
          
          cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

          cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
#
          cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)

          #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)
          cv2.circle(image_main,(self.new_center_x,self.new_center_y), 1, (255, 0, 0), -1)

          # self.top_left_x = e1[0]-self.distance_click1
          # self.top_left_y = e1[1]-self.distance_click1
          # self.bottom_right_x =e1[0]+self.distance_click1
          # self.bottom_right_Y =e1[1]+self.distance_click1


          self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
          self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
          self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
          self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
   
          self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
          self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
          self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
          self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

          self.top_left4 = e2[0]-self.distance_click2
          self.bottom_left4 =e2[1] -self.distance_click2
          self.top_right4 = e2[0]+self.distance_click2
          self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

          self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
          self.bottom_left =self.new_center_y -self.distance_click2#center_y-2#y1
          self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
          self.bottom_right = self.new_center_x +self.distance_click2#center_y+2#y2

          self.top_left2 = e3[0]-8#center_x-2#x1
          self.bottom_left2 =e3[1] -8#center_y-2#y1
          self.top_right2 = e3[0]+8#center_x+2#x2
          self.bottom_right2 =e3[1] +8#center_y+2#y2

          self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
          self.bottom_left =self.new_center_y-self.distance_click2#center_y-2#y1
          self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
          self.bottom_right = self.new_center_y +self.distance_click2#center_y+2#y2

          self.top_left5 = e4[0]-self.distance_click2
          self.bottom_left5 =e4[1] -self.distance_click2
          self.top_right5 = e4[0]+self.distance_click2
          self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 

          resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
          img_height, img_width, img_channel = resized_img3.shape
          q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
          pixmap2 = QPixmap.fromImage(q_image2)
          self.La_Srceen2.setPixmap(pixmap2)
       elif self.top_left4 <= self.x <= self.top_right4 and self.bottom_left4 <= self.y<= self.bottom_right4:
            #print("444")
              self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))#dau ngon tay tro
             
              x1=self.x1
              x2=self.x2
              x3=self.x3
              x4=self.x4
              x4[0]=x4[0]-(x4[0]-self.x)
              x3[0]=x3[0]-(x3[0]-self.x)

              self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
              self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
              self.new_center_x= int(x2[0]+self.new_width//2)
              self.new_center_y =int(x2[1]+self.new_height//2)

              
 
              e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
              e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
              e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
              e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)

              cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0),self.thickness)
              cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), self.thickness)
              cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), self.thickness)
              cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), self.thickness) 
              
              cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

              cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
    #
              cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
              cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)

              #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)
              cv2.circle(image_main,(self.new_center_x,self.new_center_y), 1, (255, 0, 0), -1)

              # self.top_left_x = e1[0]-self.distance_click1
              # self.top_left_y = e1[1]-self.distance_click1
              # self.bottom_right_x =e1[0]+self.distance_click1
              # self.bottom_right_Y =e1[1]+self.distance_click1


              self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
              self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
              self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
              self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
       
              self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
              self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
              self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
              self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

              self.top_left4 = e2[0]-self.distance_click2
              self.bottom_left4 =e2[1] -self.distance_click2
              self.top_right4 = e2[0]+self.distance_click2
              self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

              self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
              self.bottom_left =self.new_center_y -self.distance_click2#center_y-2#y1
              self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
              self.bottom_right = self.new_center_x +self.distance_click2#center_y+2#y2

              self.top_left2 = e3[0]-8#center_x-2#x1
              self.bottom_left2 =e3[1] -8#center_y-2#y1
              self.top_right2 = e3[0]+8#center_x+2#x2
              self.bottom_right2 =e3[1] +8#center_y+2#y2

              self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
              self.bottom_left =self.new_center_y-self.distance_click2#center_y-2#y1
              self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
              self.bottom_right = self.new_center_y +self.distance_click2#center_y+2#y2

              self.top_left5 = e4[0]-self.distance_click2
              self.bottom_left5 =e4[1] -self.distance_click2
              self.top_right5 = e4[0]+self.distance_click2
              self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 

              resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
              img_height, img_width, img_channel = resized_img3.shape
              q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
              pixmap2 = QPixmap.fromImage(q_image2)
              self.La_Srceen2.setPixmap(pixmap2)            
       elif self.top_left5 <= self.x <= self.top_right5 and self.bottom_left5<= self.y<= self.bottom_right5:
          #print("e4")
          # self.x_mouse=event.x()
          # self.y_mouse=event.y()
          # self.x= self.x_mouse//1.281
          # self.y= self.y_mouse//1.4
          self.La_Srceen2.setCursor(QtGui.QCursor(QtCore.Qt.SizeVerCursor))#dau mui ten doc

          x1=self.x1
          x2=self.x2
          x3=self.x3
          x4=self.x4
          #print(type(self.x1))
          #x1,x2,x3,x4 =a[0]
          x1[1]=x1[1]-(x1[1]-self.y)
          x4[1]=x4[1]-(x4[1]-self.y)



          e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
          e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
          e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
          e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)
                  
          # cv2.putText(image_main ,"X2: ",(x2[0]-4,x2[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X3: ",(x3[0]-4,x3[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X1: ",(x1[0]-4,x1[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
          # cv2.putText(image_main ,"X4: ",(x4[0]-4,x4[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)


          self.new_height = math.sqrt((x4[0] - x3[0])**2 + (x4[1] - x3[1])**2)#tinh chieu ronng moi
          self.new_width = math.sqrt((x3[0] - x2[0])**2 + (x3[1] - x2[1])**2)#tinh chieu cao
          self.new_center_x= int(x2[0]+self.new_width//2)
          self.new_center_y =int(x2[1]+self.new_height//2)

          cv2.line(image_main, (x1[0],x1[1]),(x2[0],x2[1]), (0, 255, 0),self.thickness)
          cv2.line(image_main, (x2[0],x2[1]),(x3[0],x3[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x3[0],x3[1]),(x4[0],x4[1]), (0, 255, 0), self.thickness)
          cv2.line(image_main, (x4[0],x4[1]),(x1[0],x1[1]), (0, 255, 0), self.thickness) 

          cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

          cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
#
          cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
          cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)

          #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)
          cv2.circle(image_main,(self.new_center_x,self.new_center_y), 1, (255, 0, 0), -1)

          # self.top_left_x = e1[0]-self.distance_click1
          # self.top_left_y = e1[1]-self.distance_click1
          # self.bottom_right_x =e1[0]+self.distance_click1
          # self.bottom_right_Y =e1[1]+self.distance_click1


          self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
          self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
          self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
          self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
   
          self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
          self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
          self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
          self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

          self.top_left4 = e2[0]-self.distance_click2
          self.bottom_left4 =e2[1] -self.distance_click2
          self.top_right4 = e2[0]+self.distance_click2
          self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

          self.top_left = self.new_center_x-self.distance_click2#center_x-2#x1
          self.bottom_left =self.new_center_y -self.distance_click2#center_y-2#y1
          self.top_right = self.new_center_x+self.distance_click2#center_x+2#x2
          self.bottom_right = self.new_center_y +self.distance_click2#center_y+2#y2
          #self.new_center_x=center_x
          self.top_left5 = e4[0]-self.distance_click2
          self.bottom_left5 =e4[1] -self.distance_click2
          self.top_right5 = e4[0]+self.distance_click2
          self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 
          
          resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
          img_height, img_width, img_channel = resized_img3.shape
          q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
          pixmap2 = QPixmap.fromImage(q_image2)
          self.La_Srceen2.setPixmap(pixmap2) 
         #self.La_Srceen2.mouseReleaseEvent = self.mouse_release_event 
    def Reload_ROI(self): 
        while not self.static_roi:
            #cv2.setMouseCallback('image', self.mouseEvent)
            try:
                with self.follow_lock:
                    while not self.static_roi2:
                        self.follow_lock.release()
                        time.sleep(0.05)
                        self.follow_lock.acquire()
                    
                        #print("Đang chạy")
                       #print(self.static_roi2)
  
                    while self.communication_flag:
                        self.follow_lock.release()
                        time.sleep(3)
                        self.follow_lock.acquire()

            except Exception as e:
                print(str(e))
    def Creat_ROI(self):
       image_main = self.image_main.copy()
       #self.thickness = int(self.thickness)
       print(self.resolution_camera)
       if self.resolution_camera ==0:

       	 self.resolution_camera =1 
       	 self.thickness = 1
       	 self.distance =2
       	 self.distance_click1 =15
       	 self.distance_click2 =8
       	 self.distance2 =3  
         self.angle2 =0
         self.width=100
         self.height=80
         self.x_r= 150#self.x
         self.y_r=150 #self.y
         
       elif self.resolution_camera == 2 or 3:

       	 self.distance =7
       	 self.distance2 =11
       	 self.distance_click1 =35
       	 self.distance_click2 =40
         self.angle2 =0
         self.thickness = 3
         self.width=300
         self.height=250
         self.x_r= 500#self.x
         self.y_r=500 #self.y


       self.static_roi2 = False
       self.communication_flag = True
       
       center_x = int(self.x_r)
       center_y = int(self.y_r)
        #height, width = img.shape
       center_x2=int(self.x_r)
       center_y2=int(self.y_r)

       rect_center = (center_x, center_y)
       rect_size = (int(self.width), int(self.height))
         #print(self.width,self.height)
       rect_angle = self.angle2
         #print(rect_angle)
       rect_points = cv2.boxPoints((rect_center, rect_size, rect_angle))
       cv2.circle(image_main, rect_center, 1, (255, 0, 0), -1)
       cv2.polylines(image_main, [np.intp(rect_points)], True, (0, 255, 0), self.thickness)
        
       a =[np.intp(rect_points)]
       x1,x2,x3,x4 =a[0]
       self.top_left = center_x-self.distance_click2#center_x-2#x1
       self.bottom_left =center_y -self.distance_click2#center_y-2#y1
       self.top_right = center_x+self.distance_click2#center_x+2#x2
       self.bottom_right = center_y +self.distance_click2#center_y+2#y2

       self.new_center_x=center_x
       self.new_center_y=center_y
       self.x2=x2
       self.x4=x4
       self.x1=x1
       self.x3=x3

       e1 = ((x1[0] + x2[0]) // 2, (x1[1] + x2[1]) // 2)
       e2 = ((x3[0] + x4[0]) // 2, (x3[1] + x4[1]) // 2)
       e3 = ((x2[0] + x3[0]) // 2, (x2[1] + x3[1]) // 2)
       e4 = ((x1[0] + x4[0]) // 2, (x1[1] + x4[1]) // 2)

       # self.top_left_x = e1[0]-self.distance_click1
       # self.top_left_y = e1[1]-self.distance_click1
       # self.bottom_right_x =e1[0]+self.distance_click1
       # self.bottom_right_Y =e1[1]+self.distance_click1
       #cv2.circle(image_main,(e4), 7, (0, 0, 255), 7)

       self.top_left2 = e3[0]-self.distance_click2#center_x-2#x1
       self.bottom_left2 =e3[1] -self.distance_click2#center_y-2#y1
       self.top_right2 = e3[0]+self.distance_click2#center_x+2#x2
       self.bottom_right2 =e3[1] +self.distance_click2#center_y+2#y2
       
       self.top_left3 = e1[0]-self.distance_click2#center_x-2#x1
       self.bottom_left3 =e1[1] -self.distance_click2#center_y-2#y1
       self.top_right3 = e1[0]+self.distance_click2#center_x+2#x2
       self.bottom_right3 =e1[1] +self.distance_click2#center_y+2#y2

       self.top_left4 = e2[0]-self.distance_click2
       self.bottom_left4 =e2[1] -self.distance_click2
       self.top_right4 = e2[0]+self.distance_click2
       self.bottom_right4 =e2[1] +self.distance_click2# cho goc xoay 

       self.top_left5 = e4[0]-self.distance_click2
       self.bottom_left5 =e4[1] -self.distance_click2
       self.top_right5 = e4[0]+self.distance_click2
       self.bottom_right5 =e4[1] +self.distance_click2# cho goc xoay 

       d1=x1[0]
       self.e2 =e2
       self.e1 =e1
       self.e3 =e3
       # cv2.putText(image_main ,"X2: "+str(self.angle2),(x2[0]-4,x2[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
       # cv2.putText(image_main ,"X4: "+str(self.angle2),(x4[0]-4,x4[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
       # cv2.putText(image_main ,"X1: "+str(self.angle2),(x1[0]-4,x1[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
       # cv2.putText(image_main ,"X3: "+str(self.angle2),(x3[0]-4,x3[1]-4) ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),1)
       cv2.rectangle(image_main,(x1[0]+self.distance,x1[1]+self.distance), (x1[0]-self.distance,x1[1]-self.distance), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (x2[0]-self.distance,x2[1]-self.distance), (x2[0]+self.distance,x2[1]+self.distance), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (x3[0]-self.distance,x3[1]-self.distance), (x3[0]+self.distance,x3[1]+self.distance), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (x4[0]-self.distance,x4[1]-self.distance), (x4[0]+self.distance,x4[1]+self.distance), (255, 0, 0), self.thickness)

       cv2.rectangle(image_main, (self.new_center_x-self.distance2,self.new_center_y-self.distance2), (self.new_center_x+self.distance2,self.new_center_y+self.distance2), (255, 0, 0), self.thickness)
    #
       cv2.rectangle(image_main, (e1[0]-self.distance2,e1[1]-self.distance2), (e1[0]+self.distance2,e1[1]+self.distance2), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (e2[0]-self.distance2,e2[1]-self.distance2), (e2[0]+self.distance2,e2[1]+self.distance2), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (e3[0]-self.distance2,e3[1]-self.distance2), (e3[0]+self.distance2,e3[1]+self.distance2), (255, 0, 0), self.thickness)
       cv2.rectangle(image_main, (e4[0]-self.distance2,e4[1]-self.distance2), (e4[0]+self.distance2,e4[1]+self.distance2), (255, 0, 0), self.thickness)

       #cv2.circle(image_main, e2, 4, (255, 0, 0), 2)

       #self.e2 =e2 
       self.rect_center = rect_center# cho goc xoay


  


       resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
       img_height, img_width, img_channel = resized_img3.shape
       q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
       pixmap2 = QPixmap.fromImage(q_image2)
       self.La_Srceen2.setPixmap(pixmap2)
       self.last_cenetrx=self.x_r
       self.last_cenetry=self.y_r
     

   # @jit(nopython=True)


     
    def load_image(self):
        x2_x, x2_y = self.x2[0], self.x2[1]
        x4_x, x4_y = self.x4[0], self.x4[1]
        img = self.image_main[x2_y:x4_y, x2_x:x4_x]

        self.static_roi2 = True
        #img = cv2.imread(image_path)
        self.img3= img
        #print(img)
        # # Thay đổi kích thước ảnh bằng OpenCV
        height, width = img.shape[:2] 
        self.height = height
        self.width  =width
        center = (width/2, height/2) 
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1) 
        rotated_image =cv2.warpAffine(img, rotate_matrix, (width, height))
        img5 = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        resized_img2= cv2.resize(rotated_image, (231, 231), interpolation=cv2.INTER_AREA)
        img_height, img_width, img_channel = resized_img2.shape
        q_image = QImage(
                resized_img2.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888
            )
        pixmap = QPixmap.fromImage(q_image)
        self.La_Srceen1.setPixmap(pixmap)
    def load_image2(self):
        image_path2, _ = QFileDialog.getOpenFileName(
            None,
            "Chọn ảnh",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        self.Object_count=0
        print("reset Object_count")
        if image_path2:

            #
            self.image_main = cv2.imread(image_path2)
            image_main = self.image_main
            image_shape = image_main.shape
            width = image_shape[1]
            height = image_shape[0]
            self.width_camera =width
            self.height_camera=height
            self.resolution_camera = (width*height)*0.000001
            self.resolution_camera=int(self.resolution_camera)
            resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
            img_height, img_width, img_channel = resized_img3.shape
            q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
            pixmap2 = QPixmap.fromImage(q_image2)
            self.La_Srceen2.setPixmap(pixmap2)
        
    def update_slider_value(self, value):
        #start_time = time.time()
         #value =self.value
         #self.La_Threshold.setText("0.4")
         text = self.La_Threshold.text()
         Number_To_Find=self.La_NumbertoFind.text()
         Number_To_Find= int(Number_To_Find)
         #print(Number_To_Find)
         threshold= float(text)
         if self.Object_count<Number_To_Find:
             
             #print(self.Object_count)
             #print(threshold)
             self.value = value
             self.La_Angleplus_2.setText(str(self.value))
             image_main = self.image_main
             img = self.img3
             height, width = img.shape[:2] 
             
             center = (width / 2, height / 2) 
             rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-value, scale=1.0)  # Sử dụng scale=1.0
             rotated_image = cv2.warpAffine(img, rotate_matrix, (width, height))
             img5 = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
             result = cv2.matchTemplate(cv2.cvtColor(image_main, cv2.COLOR_BGR2GRAY), img5, cv2.TM_CCOEFF_NORMED)
             #height, width = img.shape
            # Tìm vị trí khớp tốt nhất
             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
             #self.max_value2.append(max_val)
             #max_value = max(self.max_value2)
             #print(max_value)
             #max_value = max(my_list)
             #print(max_val, value)
             #threshold = 0.85 # Đặt ngưỡng ở đây, bạn có thể điều chỉnh giá trị theo nhu cầu của mình
             if max_val > threshold:
                 #self.i = self.i+1
                 #print(self.i)
                 self.Object_count=self.Object_count+1
                # Vẽ một hình chữ nhật xung quanh khu vực sự khớp tốt nhất
                 top_left = (max_loc[0], max_loc[1])
                 x, y = max_loc
                 #scale=1 
                 angle=value
                 center_x = int(x + (self.width/2))
                 center_y = int(y + (self.height/2))
                 #height, width = img.shape
                 
                 bottom_right = (top_left[0] + img5.shape[1], top_left[1] + img5.shape[0])
                 #cv2.rectangle(image_main, top_left, bottom_right, (0, 0, 255), 2)
                 #angle=value
                 #print(center_x,center_y)
                 # tính toán tọa độ các điểm góc của hình chữ nhật
                 rect_center = (center_x, center_y)
                 rect_size = (int(self.width), int(self.height))
                 #print(self.width,self.height)
                 rect_angle = angle
                 #print(rect_angle)
                 rect_points = cv2.boxPoints((rect_center, rect_size, rect_angle))
                 cv2.circle(image_main, rect_center, 1, (255, 0, 0), -1)
                 # vẽ hình chữ nhật bằng cv2.polylines
                 #print(rect_center)
                 # cv2.putText(image_main, "Object Total:"+str(self.i), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                 #cv2.polylines(image_main, [np.int0(rect_points)], True, (0, 0, 255), 1)
                 cv2.polylines(image_main, [np.intp(rect_points)], True, (0, 0, 255), 1)

                 cv2.putText(image_main ,"Angle:"+str(angle),rect_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),1)
             #cv2.putText(image_main ,"Angle:"+str(self.i),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
             resized_img3 = cv2.resize(image_main, (821,671), interpolation=cv2.INTER_AREA)
             img_height, img_width, img_channel = resized_img3.shape
             q_image2 = QImage(resized_img3.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
             pixmap2 = QPixmap.fromImage(q_image2)
             self.La_Srceen2.setPixmap(pixmap2)
             
             resized_img2= cv2.resize(rotated_image, (231, 231), interpolation=cv2.INTER_AREA)  # Không thay đổi kích thước của ảnh sau khi xoay
             img_height, img_width, img_channel = resized_img2.shape
             q_image = QImage(resized_img2.data, img_width, img_height, img_width * img_channel, QImage.Format.Format_RGB888)
             pixmap = QPixmap.fromImage(q_image)
             self.La_Srceen1.setPixmap(pixmap)
             self.Le_Total_Object.setText(str(self.Object_count))
         # end_time = time.time()
         # processing_time = (end_time - start_time)*1000
         # processing_time = f"{processing_time:.2f}"
         #print(processing_time)
    def update_slider_value2(self, value):
            #start_time = time.time()
            self.value2 = value
            self.La_Angleplus.setText(str(self.value2))
    def update_slider_value3(self, value):
            #start_time = time.time()
            self.value3 = value
            self.La_Anglemibus.setText(str(-self.value3))

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
     
