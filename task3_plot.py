import cv2
import numpy as np
import telnetlib
import time
import matplotlib.pyplot as plt
import pandas as pd
from csv import writer
import os

host1 = "192.168.0.175"    #SURAJ
host2 = "192.168.0.141"    #TECHMEET
# host = "192.168.0.141" 
port = "23"

# tn1 = telnetlib.Telnet(host1, port)
# tn2 = telnetlib.Telnet(host2, port)

# Define the range of red color in HSV
lower_red = np.array([136, 87, 111])
upper_red = np.array([180, 255, 255])

# Define the range of green color in HSV 235,228,138 : 231,231,184
green_lower = np.array([50, 100, 160], np.uint8) #72
green_upper = np.array([200, 200, 250], np.uint8)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
print(size)
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 15.0, size)

global error_1z, error_1z_prev, z_1dot, z_1prev
error_1z      = 0.0
error_1z_prev = 0.0
z_1dot        = 0.0
z_1prev       = 0.0

global error_2z, error_2z_prev, z_2dot, z_2prev
error_2z      = 0.0
error_2z_prev = 0.0
z_2dot        = 0.0
z_2prev       = 0.0

global z11,z12,z13,z14,z15,z16,z17,z18,z19,z110
z11,z12,z13,z14,z15  = 0.0,0.0,0.0,0.0,0.0
z16,z17,z18,z19,z110 = 0.0,0.0,0.0,0.0,0.0

global z21,z22,z23,z24,z25,z26,z27,z28,z29,z210
z21,z22,z23,z24,z25  = 0.0,0.0,0.0,0.0,0.0
z26,z27,z28,z29,z210 = 0.0,0.0,0.0,0.0,0.0


global z11dot,z12dot,z13dot,z14dot,z15dot
z11dot,z12dot,z13dot,z14dot,z15dot = 0.0,0.0,0.0,0.0,0.0
global z21dot,z22dot,z23dot,z24dot,z25dot
z21dot,z22dot,z23dot,z24dot,z25dot = 0.0,0.0,0.0,0.0,0.0

global error_1x, error_1x_prev
error_1x      = 0.0
error_1x_prev = 0.0

global error_2x, error_2x_prev
error_2x      = 0.0
error_2x_prev = 0.0


global x11,x12,x13,x14,x15
x11,x12,x13,x14,x15 = 0.0,0.0,0.0,0.0,0.0

global x21,x22,x23,x24,x25
x21,x22,x23,x24,x25 = 0.0,0.0,0.0,0.0,0.0

global x11dot,x12dot,x13dot,x14dot,x15dot
x11dot,x12dot,x13dot,x14dot,x15dot = 0.0,0.0,0.0,0.0,0.0

global x21dot,x22dot,x23dot,x24dot,x25dot
x21dot,x22dot,x23dot,x24dot,x25dot = 0.0,0.0,0.0,0.0,0.0

global error_1y, error_1y_prev
error_1y      = 0.0
error_1y_prev = 0.0

global error_2y, error_2y_prev
error_2y      = 0.0
error_2y_prev = 0.0

global y11,y12,y13,y14,y15
y11,y12,y13,y14,y15 = 0.0,0.0,0.0,0.0,0.0

global y21,y22,y23,y24,y25
y21,y22,y23,y24,y25 = 0.0,0.0,0.0,0.0,0.0

global y11dot,y12dot,y13dot,y14dot,y15dot
y11dot,y12dot,y13dot,y14dot,y15dot = 0.0,0.0,0.0,0.0,0.0

global y21dot,y22dot,y23dot,y24dot,y25dot
y21dot,y22dot,y23dot,y24dot,y25dot = 0.0,0.0,0.0,0.0,0.0

filename_to_log_drone_data = "data.csv"
try:
    os.remove(filename_to_log_drone_data)
except:
    pass

def distfunc(x1,y1,x2,y2):
    l = ((x2-x1)**2+(y2-y1)**2)**0.5
    return l

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
    # print(tn.read_some())
    return bytes(cmd1)

def set(data):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, data, 0])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)

def path_xy(x1,y1,x2,y2,t,ti,tf):
    t = t-ti
    if t+ti<=ti:
        x_d = x1
        y_d = y1
    elif ti<t+ti<tf:
        dt = tf-ti
        if x1==x2:
            x_d = x1
            y_d = y1 + 3*(y2-y1)*t**2/dt**2 + 2*(y1-y2)*t**3/dt**3
        if y1==y2:
            y_d = y1
            x_d = x1 + 3*(x2-x1)*t**2/dt**2 + 2*(x1-x2)*t**3/dt**3
    elif t+ti>=tf:
        x_d = x2
        y_d = y2
    
    # print(x_d,y_d)
    return int(x_d),int(y_d)


# tn1.write(rc(1500,1500,1500,1500,901,901,1500,901))
# print(tn1.read_some())
# tn1.write(rc(1500,1500,1000,1500,901,901,1500,1500))
# print(tn1.read_some())
# tn2.write(rc(1500,1500,1500,1500,901,901,1500,901))
# print(tn2.read_some())
# tn2.write(rc(1500,1500,1000,1500,901,901,1500,1500))
# print(tn2.read_some())
# tn.write(set(1))
# print(tn.read_some())

t0 = time.time()
start_log_flag_drone=1
start_vel_flag=1
t_cur=0

error_1z_int=0
error_2z_int=0

firstflag_green = 1
firstflag_red = 1
x,y=0,0


global x1_d1,y1_d1,x1_d2,y1_d2,x1_d3,y1_d3,x1_d4,y1_d4
global t10,t11,t12,t13,t14
# desired for drone1
x1_d1,y1_d1=250,70
x1_d2,y1_d2=740,70
x1_d3,y1_d3=740,306
x1_d4,y1_d4=250,306
# desired for drone2
x2_d2,y2_d2=250,70
x2_d3,y2_d3=740,70
x2_d4,y2_d4=740,306
x2_d1,y2_d1=250,306

# time for drone1
t_10=10
t11=30
t12=40
t13=60
t14=70

# time for drone2
t_20=10
t21=30
t22=40
t23=60
t24=70

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for red color
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Create a mask for green color
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find contours in red mask
    red_contours, red_hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in green mask
    green_contours, green_hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # x1,x2,y1,y2 = 0,0,0,0
    # w1,w2,h1,h2 = 0,0,0,0

    # x_d=510
    # y_d=245
    # z_d=35
    

    # print(t_cur)
    if(t_cur<t11):
        x_1d,y_1d=path_xy(x1_d1,y1_d1,x1_d2,y1_d2,t_cur,t_10,t11)
    elif(t11<t_cur<t12):
        x_1d,y_1d=path_xy(x1_d2,y1_d2,x1_d3,y1_d3,t_cur,t11,t12)
    elif(t12<t_cur<t13):
        x_1d,y_1d=path_xy(x1_d3,y1_d3,x1_d4,y1_d4,t_cur,t12,t13)
    elif(t13<t_cur<t14):
        x_1d,y_1d=path_xy(x1_d4,y1_d4,x1_d1,y1_d1,t_cur,t13,t14)
    z_1d=35

    if(t_cur<t21):
        x_2d,y_2d=path_xy(x2_d1,y2_d1,x2_d2,y2_d2,t_cur,t_20,t21)
    elif(t21<t_cur<t22):
        x_2d,y_2d=path_xy(x2_d2,y2_d2,x2_d3,y2_d3,t_cur,t21,t22)
    elif(t22<t_cur<t23):
        x_2d,y_2d=path_xy(x2_d3,y2_d3,x2_d4,y2_d4,t_cur,t22,t23)
    elif(t23<t_cur<t24):
        x_2d,y_2d=path_xy(x2_d4,y2_d4,x2_d1,y2_d1,t_cur,t23,t24)
    z_2d=35

    # x_d,y_d = path_xy(x_d1,y_d1,x_d2,y_d2,t_cur,5,15)
    # z_d     = 30
    z_1d_dot = 0.0
    z_2d_dot = 0.0
    # z_d=25
    # if(t_cur<5):
    #     x_d=249
    #     y_d=38
    # elif (x_d<550 and y_d==38):
    #     x_d+=int(1*(t_cur-5))
    #     y_d=38
    # elif (y_d<339 and x_d==552):
    #     x_d=550
    #     y_d+=int(1*(t_cur-5))
    # elif(x_d>249 and y_d==339):
    #     x_d-=int(1*(t_cur-5))
    #     y_d=339
    # elif(y_d<38 and x_d==249):
    #     x_d=249
    #     y_d+=int(1*(t_cur-5))
    # print(x_d,y_d)
    
    # else:
        # x_d=460
    
    area1,area2=0,0
    x1,x2,y1,y2=0,0,0,0
    xr,yr,wr,hr=x1,y1,0,0
    xg,yg,wg,hg=x2,y2,0,0
    
    if len(red_contours) > 0:
        # enter for first time
        if firstflag_red == 1:
            
            firstflag_red = 2
            # Find the largest contour in red contours
            red_c = max(red_contours, key=cv2.contourArea)

            # Draw a bounding box around the red contour
            xr, yr, wr, hr = cv2.boundingRect(red_c)
            cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)

            # Calculate the center of the bounding box
            center_OG = (xr + int(wr/2), yr + int(hr/2))
            center_OG2 = np.array([xr + int(wr/2), yr + int(hr/2)])

            # Put the coordinates on the bounding box
            cv2.putText(frame, str(center_OG), (xr, yr-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Find the closest contour to OG_red in green contours
        meanlist = []
        for i in red_contours:
            meanlist.append(np.array([np.mean(i[:,0,0]),np.mean(i[:,0,1]),i]))
        dist = 10000
        for j in meanlist:
            
            if dist > np.linalg.norm(j[0:2]-center_OG2):
                dist = np.linalg.norm(j[0:2]-center_OG2)
                new_center_r = j[0:2]
                new_contour_r = j[2]
        
        center_OG2 = new_center_r
        area1=cv2.contourArea(new_contour_r)
            
        
        # green_c = max(green_contours, key=cv2.contourArea)

        # Draw a bounding box around the green contour
        xr, yr, wr, hr = cv2.boundingRect(new_contour_r)
        cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)

        # Calculate the center of the bounding box
        center_r = (xr + int(wr/2), yr + int(hr/2))

        # Put the coordinates on the bounding box
        cv2.putText(frame, str(center_r), (xr, yr-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if len(green_contours) > 0:
        # enter for first time

        if firstflag_green == 1:
            
            firstflag_green = 2
                    # Find the largest contour in green contours
            green_c = max(green_contours, key=cv2.contourArea)

            # Draw a bounding box around the green contour
            xg, yg, wg, hg = cv2.boundingRect(green_c)
            cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

            # Calculate the center of the bounding box
            center_OG = (xg + int(wg/2), yg + int(hg/2))
            center_OG2 = np.array([xg + int(wg/2), yg + int(hg/2)])

        
            # Put the coordinates on the bounding box
            cv2.putText(frame, str(center_OG), (xg, yg-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Find the closest contour to OG_green in green contours
        # print(green_contours)
        # print(np.shape(green_contours))
        meanlist = []
        for i in green_contours:
            meanlist.append(np.array([np.mean(i[:,0,0]),np.mean(i[:,0,1]),i]))
        dist = 10000
        for j in meanlist:
            
            if dist > np.linalg.norm(j[0:2]-center_OG2):
                dist = np.linalg.norm(j[0:2]-center_OG2)
                new_center_g = j[0:2]
                new_contour_g = j[2]
        
        center_OG2 = new_center_g
        area2=cv2.contourArea(new_contour_g)
            
        
        # green_c = max(green_contours, key=cv2.contourArea)

        # Draw a bounding box around the green contour
        xg, yg, wg, hg = cv2.boundingRect(new_contour_g)
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

        # Calculate the center of the bounding box
        center = (xg + int(wg/2), yg + int(hg/2))
        # Put the coordinates on the bounding box
        cv2.putText(frame, str(center), (xg, yg-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    x1 = xr
    y1 = yr
    z1=area1/10

    x2 = xg
    y2 = yg
    z2=area2/10
    # print(area1,area2)

    ################## for x

    x15 = x14
    x14 = x13
    x13 = x12
    x12 = x11
    x11 = x1
    x1  = (x11+x12+x13+x14+x15)/5

    x25 = x24
    x24 = x23
    x23 = x22
    x22 = x21
    x21 = x2
    x2  = (x21+x22+x23+x24+x25)/5
    
    error_1x     = x_1d-x1
    error_1x_dot = error_1x-error_1x_prev

    error_2x     = x_2d-x2
    error_2x_dot = error_2x-error_2x_prev

    x15dot = x14dot
    x14dot = x13dot
    x13dot = x12dot
    x12dot = x11dot
    x11dot = error_1x_dot
    error_1x_dot = (x11dot+x12dot+x13dot+x14dot+x15dot)/5
    x25dot = x24dot
    x24dot = x23dot
    x23dot = x22dot
    x22dot = x21dot
    x21dot = error_2x_dot
    error_2x_dot = (x21dot+x22dot+x23dot+x24dot+x25dot)/5
    
    error_1x_prev=error_1x
    error_2x_prev=error_2x

    if(x1<0):
        x1=0
    if(x1>854):
        x1=854
    if(x2<0):
        x2=0
    if(x2>854):
        x2=854
    ################## for y

    y15 = y14
    y14 = y13
    y13 = y12
    y12 = y11
    y11 = y1
    y1  = (y11+y12+y13+y14+y15)/5

    y25 = y24
    y24 = y23
    y23 = y22
    y22 = y21
    y21 = y2
    y2  = (y21+y22+y23+y24+y25)/5
    
    error_1y     = y_1d-y1
    error_1y_dot = error_1y-error_1y_prev

    error_2y     = y_2d-y2
    error_2y_dot = error_2y-error_2y_prev

    y15dot = y14dot
    y14dot = y13dot
    y13dot = y12dot
    y12dot = y11dot
    y11dot = error_1y_dot
    error_1y_dot = (y11dot+y12dot+y13dot+y14dot+y15dot)/5
    y25dot = y24dot
    y24dot = y23dot
    y23dot = y22dot
    y22dot = y21dot
    y21dot = error_2y_dot
    error_2y_dot = (y21dot+y22dot+y23dot+y24dot+y25dot)/5
    
    error_1y_prev=error_1y
    error_2y_prev=error_2y

    if(y1<0):
        y1=0
    if(y1>854):
        y1=854
    if(y2<0):
        y2=0
    if(y2>854):
        y2=854

    # z1=np.round((wr**2+hr**2)**0.5,2)
    # z2=np.round((w2**2+h2**2)**0.5,2)

    ################## for z

    # z19=z18
    # z18=z17
    # z17=z16
    # z16=z15
    # z15=z14
    # z14=z13
    # z13=z12
    # z12=z11
    # z11=z10

    z110=z19
    z19=z18
    z18=z17
    z17=z16
    z16=z15
    z15=z14
    z14=z13
    z13=z12
    z12=z11
    z11=z1

    z=(z11+z12+z13+z14+z15)/5

    if(z1<0):
        z1=0
    if(z1>50):
        z1=50

    z210=z29
    z29=z28
    z28=z27
    z27=z26
    z26=z25
    z25=z24
    z24=z23
    z23=z22
    z22=z21
    z21=z2

    z=(z21+z22+z23+z24+z25)/5

    if(z2<0):
        z2=0
    if(z2>50):
        z2=50


    error_1z = z_1d-z1
    error_1z_dot=z_1d_dot - z_1dot
    z_1dot = (z1 - z_1prev)
    error_1z_int += error_1z

    z15dot=z14dot
    z14dot=z13dot
    z13dot=z12dot
    z12dot=z11dot
    z11dot=z_1dot

    # z_dot=(5*z1dot+4*z2dot+3*z3dot+2*z4dot+1*z5dot)/(5+4+3+2+1)

    z_1prev=z1

    error_2z = z_2d-z2
    error_2z_dot=z_2d_dot - z_2dot
    z_2dot = (z2 - z_2prev)
    error_2z_int += error_2z

    z25dot=z24dot
    z24dot=z23dot
    z23dot=z22dot
    z22dot=z22dot
    z22dot=z_2dot

    # z_dot=(5*z2dot+4*z2dot+3*z3dot+2*z4dot+2*z5dot)/(5+4+3+2+2)

    z_2prev=z2

    cap_x=200

    if(error_1x_dot<-cap_x):
        error_1x_dot=-cap_x
    if(error_1x_dot>cap_x):
        error_1x_dot=cap_x

    if(error_2x_dot<-cap_x):
        error_2x_dot=-cap_x
    if(error_2x_dot>cap_x):
        error_2x_dot=cap_x


    cap_y=200

    if(error_1y_dot<-cap_y):
        error_1y_dot=-cap_y
    if(error_1y_dot>cap_y):
        error_1y_dot=cap_y

    if(error_2y_dot<-cap_y):
        error_2y_dot=-cap_y
    if(error_2y_dot>cap_y):
        error_2y_dot=cap_y

    kp_z=15
    kd_z=350    #365-456
    ki_z=0.0
    throttle_offset=50


    kp_x=1
    kd_x=20
    
    kp_y=1
    kd_y=20
    kp_zx = 0

    if(t_cur<5):
        print("inside")
        kp_z=7
        kp_x=0
        kd_x=0
        kp_y=0
        kd_y=0
        kp_zx = 0
    
    pitch1 = int(1500 +12+ kp_x*error_1x + kd_x*error_1x_dot)
    roll1 = int(1500 + kp_y*error_1y + kd_y*error_1y_dot)
    throttle1 = int(1500 + throttle_offset + kp_z*error_1z + kd_z*error_1z_dot + ki_z*error_1z_int + kp_zx*error_1x)
    
    pitch2 = int(2500 +12+ kp_x*error_2x + kd_x*error_2x_dot)
    roll2 = int(2500 + kp_y*error_2y + kd_y*error_2y_dot)
    throttle2 = int(2500 + throttle_offset + kp_z*error_2z + kd_z*error_2z_dot + ki_z*error_2z_int + kp_zx*error_2x)

    if(pitch1>1550):
        pitch1=1550
    elif(pitch1<1450):
        pitch1=1450

    if(roll1>1550):
        roll1=1550
    elif(roll1<1450):
        roll1=1450

    if(pitch2>1550):
        pitch2=1550
    elif(pitch2<1450):
        pitch2=1450

    if(roll2>1550):
        roll2=1550
    elif(roll2<1450):
        roll2=1450
        
    if(start_vel_flag==1):
        throttle2=1500
        start_vel_flag=0
    if(start_vel_flag==1):
        throttle1=1500
        start_vel_flag=0

    
    t_cur = time.time() - t0

    print("e1z, ", np.round(kp_z*error_1z,1), "ex_1d", np.round(kd_z*error_1z_dot,1) ,end=" ")
    print("throttle1: ", throttle1)

    print("e2z, ", np.round(kp_z*error_2z,1), "ex_2d", np.round(kd_z*error_2z_dot,1) ,end=" ")
    print("throttle2: ", throttle2)

    print()

    ##### log flight data for intruder drone
    
    header = ['time','z','z_dot']
    with open(filename_to_log_drone_data, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        data_ = [float(t_cur), float(error_1z), float(error_1z_dot), float(error_2z), float(error_2z_dot)]
        if start_log_flag_drone == 1:
            writer_object.writerow(header)
            start_log_flag_drone = 0
        else:
            writer_object.writerow(data_)
            f_object.close()

    # print("fps:",cap.get(cv2.CAP_PROP_FPS),end=" ")
    # print("position:", (x,y,z))

    # tn1.write(rc(roll1,pitch1,throttle1,1500,900,900,900,900))
    # tn2.write(rc(roll2,pitch2,throttle2,1500,900,900,900,900))

    # Display the resulting frame
    frame = cv2.circle(frame, (x_1d,y_1d), 7, (255,0,0), -1)
    frame = cv2.circle(frame, (x_2d,y_2d), 7, (0,0,255), -1)
    cv2.imshow("Frame", frame)
    out.write(frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # tn1.write(set(2))
        # print(tn1.read_some())
        # time.sleep(2)
        # tn1.write(rc(1500,1500,1500,1500,900,900,900,900))

        # tn2.write(set(2))
        # print(tn2.read_some())
        # time.sleep(2)
        # tn2.write(rc(1500,1500,1500,1500,900,900,900,900))

        break
    # print(dt)

cap.release()
out.release()
cv2.destroyAllWindows()
# tn1.close()
# tn2.close()
#### Final ploting code for intruder drone

Saved_data = pd.read_csv('data.csv')

plt.figure(1)

# plt.subplot(3,2,1)
plt.plot(Saved_data['time'], Saved_data['z'])
plt.plot(Saved_data['time'], Saved_data['z_dot'])
plt.legend("z","z dot")
plt.grid()

plt.show()
