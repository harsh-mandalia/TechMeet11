import cv2
import numpy as np
import telnetlib
import time
import matplotlib.pyplot as plt
import pandas as pd
from csv import writer
import os

host = "192.168.4.1"
# host = "192.168.0.141" 
port = "23"

# tn = telnetlib.Telnet(host, port)

# Define the range of red color in HSV
lower_red = np.array([136, 87, 111])
upper_red = np.array([180, 255, 255])

# Define the range of green color in HSV
green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

global error_z, error_z_prev, z_dot, z_prev
error_z=0.0
error_z_prev=0.0
z_dot = 0.0
z_prev = 0.0

global z1,z2,z3,z4,z5,z6,z7,z8,z9,z10
z1,z2,z3,z4,z5=0.0,0.0,0.0,0.0,0.0
z6,z7,z8,z9,z10=0.0,0.0,0.0,0.0,0.0

global z1dot,z2dot,z3dot,z4dot,z5dot
z1dot,z2dot,z3dot,z4dot,z5dot=0.0,0.0,0.0,0.0,0.0

global error_x, error_x_prev
error_x=0.0
error_x_prev=0.0

global x1,x2,x3,x4,x5
x1,x2,x3,x4,x5=0.0,0.0,0.0,0.0,0.0

global x1dot,x2dot,x3dot,x4dot,x5dot
x1dot,x2dot,x3dot,x4dot,x5dot=0.0,0.0,0.0,0.0,0.0

global error_y, error_y_prev
error_y=0.0
error_y_prev=0.0

global y1,y2,y3,y4,y5
y1,y2,y3,y4,y5=0.0,0.0,0.0,0.0,0.0

global y1dot,y2dot,y3dot,y4dot,y5dot
y1dot,y2dot,y3dot,y4dot,y5dot=0.0,0.0,0.0,0.0,0.0

filename_to_log_drone_data = "data.csv"
try:
    os.remove(filename_to_log_drone_data)
except:
    pass

def dist(x1,y1,x2,y2):
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
    t=t-ti
    if(t+ti<=ti):
        x_d=x1
        y_d=y1
    elif(ti<t+ti<tf):
        dt=tf-ti
        if(x1==x2):
            x_d=x1
            y_d = y1 + 3*(y2-y1)*t**2/dt**2 + 2*(y1-y2)*t**3/dt**3
        if(y1==y2):
            y_d=y1
            x_d = x1 + 3*(x2-x1)*t**2/dt**2 + 2*(x1-x2)*t**3/dt**3
    elif(t+ti>=tf):
        x_d=x2
        y_d=y2
    
    print(x_d,y_d)
    return int(x_d),int(y_d)

# tn.write(rc(1500,1500,1500,1500,901,901,1500,901))
# print(tn.read_some())
# tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
# print(tn.read_some())

t0 = time.time()
start_log_flag_drone=1
start_vel_flag=1
t_cur=0
error_z_int=0

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

    x1,x2,y1,y2=0,0,0,0
    w1,w2,h1,h2=0,0,0,0

    # x_d=510
    # y_d=245
    x_d1,y_d1=249,38
    x_d2,y_d2=550,38
    x_d3,y_d3=550,339
    x_d4,y_d4=249,339
    x_d5,y_d5=249,38

    if(t_cur<15):
        x_d,y_d=path_xy(x_d1,y_d1,x_d2,y_d2,t_cur,5,15)
    elif(15<t_cur<25):
        x_d,y_d=path_xy(x_d2,y_d2,x_d3,y_d3,t_cur,15,25)
    elif(25<t_cur<35):
        x_d,y_d=path_xy(x_d3,y_d3,x_d4,y_d4,t_cur,25,35)
    elif(35<t_cur<45):
        x_d,y_d=path_xy(x_d4,y_d4,x_d1,y_d1,t_cur,35,45)
    z_d=30
    
    # z_d=25
    # t_path=int(t_cur)
    # if(t_path<5):
    #     x_d=249
    #     y_d=38
    # elif (t_path<10):
    #     x_d+=int(1*(t_path-5))
    #     y_d=38
    # elif (t_path<15):
    #     x_d=550
    #     y_d+=int(1*(t_path-5))
    # elif(t_path<20):
    #     x_d-=int(1*(t_path-5))
    #     y_d=339
    # elif(t_path<25):
    #     x_d=249
    #     y_d+=int(1*(t_path-5))
    
    # print(x_d,y_d)
    
    # else:
        # x_d=460
    

    z_d_dot = 0.0

    if len(red_contours) > 0:
        # Find the largest contour in red contours
        red_c = max(red_contours, key=cv2.contourArea)

        # Draw a bounding box around the red contour
        x1, y1, w1, h1 = cv2.boundingRect(red_c)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

        # Calculate the center of the bounding box
        center = (x1 + int(w1/2), y1 + int(h1/2))

        # Put the coordinates on the bounding box
        cv2.putText(frame, str(center), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if len(green_contours) > 0:
        # Find the largest contour in green contours
        green_c = max(green_contours, key=cv2.contourArea)

        # Draw a bounding box around the green contour
        x2, y2, w2, h2 = cv2.boundingRect(green_c)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

        # Calculate the center of the bounding box
        center = (x2 + int(w2/2), y2 + int(h2/2))

        # Put the coordinates on the bounding box
        cv2.putText(frame, str(center), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    x=(x1+x2)/2
    y=(y1+y2)/2

    ################## for x

    x5=x4
    x4=x3
    x3=x2
    x2=x1
    x1=x

    x=(x1+x2+x3+x4+x5)/5
    
    error_x = x_d-x
    error_x_dot=error_x-error_x_prev

    x5dot=x4dot
    x4dot=x3dot
    x3dot=x2dot
    x2dot=x1dot
    x1dot=error_x_dot

    error_x_dot=(x1dot+x2dot+x3dot+x4dot+x5dot)/5

    error_x_prev=error_x

    if(x<0):
        x=0
    if(x>854):
        x=854

    ################## for y

    y5=y4
    y4=y3
    y3=y2
    y2=y1
    y1=y

    y=(y1+y2+y3+y4+y5)/5

    error_y = y_d-y
    error_y_dot=error_y-error_y_prev

    y5dot=y4dot
    y4dot=y3dot
    y3dot=y2dot
    y2dot=y1dot
    y1dot=error_y_dot

    error_y_dot=(y1dot+y2dot+y3dot+y4dot+y5dot)/5

    error_y_prev=error_y

    if(y<0):
        y=0
    if(y>854):
        y=854

    z=np.round(dist(x1,y1,x2,y2),2)

    ################## for z

    z10=z9
    z9=z8
    z8=z7
    z7=z6
    z6=z5
    z5=z4
    z4=z3
    z3=z2
    z2=z1
    z1=z

    z=(5*z1+4*z2+3*z3+2*z4+1*z5)/(5+4+3+2+1)

    if(z<0):
        z=0
    if(z>50):
        z=50

    error_z = z_d-z
    error_z_dot=z_d_dot - z_dot
    z_dot = (z - z_prev)
    error_z_int += error_z

    z5dot=z4dot
    z4dot=z3dot
    z3dot=z2dot
    z2dot=z1dot
    z1dot=z_dot

    # z_dot=(5*z1dot+4*z2dot+3*z3dot+2*z4dot+1*z5dot)/(5+4+3+2+1)
    

    z_prev=z

    cap_x=200

    if(error_x_dot<-cap_x):
        error_x_dot=-cap_x
    if(error_x_dot>cap_x):
        error_x_dot=cap_x

    cap_y=200

    if(error_y_dot<-cap_y):
        error_y_dot=-cap_y
    if(error_y_dot>cap_y):
        error_y_dot=cap_y

    kp_z=58    #57
    kd_z=500    #365-456
    ki_z=0.01
    throttle_offset=60

    kp_x=1
    kd_x=5
    
    kp_y=1
    kd_y=5
    
    pitch = int(1500 + kp_x*error_x + kd_x*error_x_dot)
    roll = int(1500 + kp_y*error_y + kd_y*error_y_dot)
    throttle = int(1500 + throttle_offset + kp_z*error_z + kd_z*error_z_dot + ki_z*error_z_int)
    
    # if(throttle>1750):
    #     throttle=1750
    # elif(throttle<1250):
    #     throttle=1250
    if(start_vel_flag==1):
        throttle=1500
        start_vel_flag=0

    
    t_cur = time.time() - t0
    # print(t_cur)

    print("ez, ", np.round(kp_z*error_z,1), "ex_d",np.round(kd_z*error_z_dot,1), "ez_int", np.round(ki_z*error_z_int,1),end=" ")
    print("throttle: ", (x,y))

    ##### log flight data for intruder drone
    
    header = ['time','z','z_dot']
    with open(filename_to_log_drone_data, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        data_ = [float(t_cur), float(error_z), float(error_z_dot)]
        if start_log_flag_drone == 1:
            writer_object.writerow(header)
            start_log_flag_drone = 0
        else:
            writer_object.writerow(data_)
            f_object.close()

    # print("fps:",cap.get(cv2.CAP_PROP_FPS),end=" ")
    # print("position:", (x,y,z))

    # tn.write(rc(roll,pitch,throttle,1500,900,900,900,900))

    # Display the resulting frame
    frame = cv2.circle(frame, (x_d,y_d), 10, (255,0,0), -1)
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # tn.write(set(2))
        # tn.write(rc(1500,1500,1500,1500,900,900,900,900))
        # print(tn.read_some())
        break
    # print(dt)

cap.release()
cv2.destroyAllWindows()
# tn.close()

#### Final ploting code for intruder drone

Saved_data = pd.read_csv('data.csv')

plt.figure(1)

# plt.subplot(3,2,1)
plt.plot(Saved_data['time'], Saved_data['z'])
plt.plot(Saved_data['time'], Saved_data['z_dot'])
plt.legend("z","z dot")
plt.grid()

plt.show()
