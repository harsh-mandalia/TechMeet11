# press "q" to close the window

import cv2 as cv
from numpy.core.numeric import tensordot
import numpy as np
import time
import telnetlib

host = "192.168.4.1"
port = "23"

tn = telnetlib.Telnet(host, port)
# print(tn.read_some())

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)
    
def rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4):
    if(roll>2100):
        roll=2100
    elif(roll<900):
        roll=900

    if(pitch>2100):
        pitch=2100
    elif(pitch<900):
        pitch=900

    if(throttle>2100):
        throttle=2100
    elif(throttle<900):
        throttle=900

    cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
    add_checksum(cmd1)
    # print(bytes(cmd1))
    return bytes(cmd1)

def set(data):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, data, 0])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)

def dist(x1,y1,x2,y2):
    l = ((x2-x1)**2+(y2-y1)**2)**0.5
    return l

#these two matrices are what you will get after calibrating your camera
cameraMatrix=np.array([[785.9682179, 0, 357.16177299], [0,792.81486091, 105.86987584], [0, 0, 1]])
distCoeffs=np.array([[0.11186678, 0.56365852, -0.06475133, 0.01037753, -1.38199825]])

cv.namedWindow("video")
cap=cv.VideoCapture(2)
# cap = cv.VideoCapture(2, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

dict=cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
size = (1920, 1080)
out = cv.VideoWriter('output2.mp4', cv.VideoWriter_fourcc(*'MJPG'), 5.0, size)

# tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
# print(tn.read_some())
tn.write(rc(1500,1500,1500,1500,901,901,1500,901))
print(tn.read_some())
tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
print(tn.read_some())
# time.sleep(0.5)
# tn.write(rc(1500,1500,1550,1500,901,901,1500,1500))
# print(tn.read_some())

t0=time.time()

while(cap.isOpened()):
    ret, frame=cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    outputFrame = frame.copy()
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(frame, dict)

    x_d=1061
    y_d=427
    z_d=22

    t=time.time()
    
    
    if len(corners)>0 and t-t0>5:
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

        pitch = int(1500+0.1*(x_d-x))
        roll = int(1500+0.1*(y_d-y))
        throttle = int(1500+1*(z_d-z))
        print(roll,pitch,throttle)

        tn.write(rc(roll,pitch,throttle,1500,901,901,1500,1500))
        
        # print("fps:",cap.get(cv.CAP_PROP_FPS),end=" ")
        
        # print("position:", (x,y,z))         #position (x,y) in terms of pixels
    else:
        tn.write(rc(1500-3,1500-2,1600,1500,901,901,1500,1500))
    
    outputFrame = cv.circle(outputFrame, (x_d,y_d), 10, (255,0,0), -1)

    cv.imshow("video",outputFrame)
    out.write(outputFrame)

    if(cv.waitKey(1)==ord('q')):
        print("frame size:",outputFrame.shape[:2])        #size of your video in pixels
        tn.write(set(2))
        print(tn.read_some())
        break
    #time.sleep(0.1)
cap.release()
out.release()
cv.destroyAllWindows()