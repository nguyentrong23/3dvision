import cv2
import pyrealsense2
from realsense_depth import *
import numpy as np

cap =  DepthCamera()
while 1:
    ret, frame3d, frame2d = cap.get_frame()
    point = (400,300)
    cv2.circle(frame2d, point, 5, (0, 255, 255))
    dis = frame3d[point[1],point[0]]
    print(dis)

    cv2.imshow("frame3d", frame3d)
    cv2.imshow("Img",frame2d)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cv2.destroyAllWindows()