import numpy as np
import cv2 as cv
import glob
import os 

def calibrate(showPics=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root,'demoImages\calibration')
    if os.path.isdir(calibrationDir):
        image_name = [f for f in os.listdir(calibrationDir) if f.endswith('.jpg')]
    nRows = 9
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []
    for name in  image_name:
        curImgPath = calibrationDir +"\\"+ name
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None)
        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()
    repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
    print(camMatrix)
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder,'calibration.npz')
    np.savez(paramPath,
        repError=repError,
        camMatrix=camMatrix,
        distCoeff=distCoeff,
        rvecs=rvecs,
        tvecs=tvecs)
    return camMatrix,distCoeff

def removeDistortion(camMatrix,distCoeff): 
    root = os.getcwd()
    imgPath = os.path.join(root,'demoImages//distortion2.jpg')
    img = cv.imread(imgPath)
    height,width = img.shape[:2]
    camMatrixNew,roi = cv.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height)) 
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)
    return imgUndist
    # cv.line(img,(1769,103),(1780,922),(255,0,255),2,cv.LINE_AA)
    # cv.line(imgUndist,(1769,103),(1780,922),(0,255,255),2,cv.LINE_AA)
    # cv.imshow('img', img)
    # cv.imshow('imgUndist', imgUndist)
    # cv.waitKey(0)

def runCalibration(): 
    calibrate(showPics=True) 

def runRemoveDistortion():
    camMatrix,distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix,distCoeff)

if __name__ == '__main__': 
    runRemoveDistortion()