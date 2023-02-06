# press "q" to close the window

from math import dist
import cv2 as cv
from numpy.core.numeric import tensordot
import numpy as np
import time

#these two matrices are what you will get after calibrating your camera
cameraMatrix=np.array([[785.9682179, 0, 357.16177299], [0,792.81486091, 105.86987584], [0, 0, 1]])
distCoeffs=np.array([[0.11186678, 0.56365852, -0.06475133, 0.01037753, -1.38199825]])

def dist(x1,y1,x2,y2):
    l = ((x2-x1)**2+(y2-y1)**2)**0.5
    return l

cv.namedWindow("video")
cap=cv.VideoCapture(2)
# cap = cv.VideoCapture(2, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FPS, 60)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

dict=cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
size = (1920, 1080)
# out = cv.VideoWriter('output2.mp4', cv.VideoWriter_fourcc(*'MJPG'), 50.0, size)

while(cap.isOpened()):
    ret, frame=cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    x_d=1061
    y_d=427
    z_d=22
    # print("fps:",cap.get(cv.CAP_PROP_FPS))

    outputFrame = frame.copy()
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(frame, dict)
    
    if len(corners)>0:
        outputFrame = cv.aruco.drawDetectedMarkers(outputFrame,corners,ids)

        x1=corners[0][0][0][0]
        y1=corners[0][0][0][1]
        x2=corners[0][0][1][0]
        y2=corners[0][0][1][1]
        x3=corners[0][0][2][0]
        y3=corners[0][0][2][1]
        x4=corners[0][0][3][0]
        y4=corners[0][0][3][1]

        l1=dist(x1,y1,x2,y2)
        l2=dist(x2,y2,x3,y3)
        l3=dist(x3,y3,x4,y4)
        l4=dist(x4,y4,x1,y1)
        
        x=(x1+x2+x3+x4)/4
        y=(y1+y2+y3+y4)/4
        z=np.round((l1+l2+l3+l4)/4,2)
        
        print("fps:",cap.get(cv.CAP_PROP_FPS),end=" ")
        
        print("position:", (x,y,z))         #position (x,y) in terms of pixels

    # outputFrame = cv.aruco.drawAxis(outputFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
    outputFrame = cv.circle(outputFrame, (x_d,y_d), 10, (255,0,0), -1)
    cv.imshow("video",outputFrame)
    # out.write(outputFrame)

    if(cv.waitKey(1)==ord('q')):
        print("frame size:",outputFrame.shape[:2])        #size of your video in pixels
        break
cap.release()
# out.release()
cv.destroyAllWindows()