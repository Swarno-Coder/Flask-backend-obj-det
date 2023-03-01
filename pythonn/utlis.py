import cv2
import numpy as np

def getContours(img, canTh=[100,100], showCanny=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur,canTh[0],canTh[1])
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThr = cv2.erode(imgDil, kernel, iterations=3)
    if showCanny:
        cv2.imshow('Canny',imgCanny)
        cv2.imshow('Dilate',imgDil)
        cv2.imshow('Erode',imgThr)