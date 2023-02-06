import cv2
import numpy as np
import telnetlib

host = "192.168.4.1" 
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

def distfunc(x1,y1,x2,y2):
    l = ((x2-x1)**2+(y2-y1)**2)**0.5
    return l

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)
    
def rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)

def set(data):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, data, 0])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)


firstflag_green = 1
firstflag_red = 1

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

    x_d=510
    y_d=245
    z_d=16

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
        # enter for first time

        if firstflag_green == 1:
            
            firstflag_green = 2
                    # Find the largest contour in green contours
            green_c = max(green_contours, key=cv2.contourArea)

            # Draw a bounding box around the green contour
            x2, y2, w2, h2 = cv2.boundingRect(green_c)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

            # Calculate the center of the bounding box
            center_OG = (x2 + int(w2/2), y2 + int(h2/2))
            center_OG2 = np.array([x2 + int(w2/2), y2 + int(h2/2)])

        
            # Put the coordinates on the bounding box
            cv2.putText(frame, str(center), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Find the closest contour to OG_green in green contours
        print(green_contours)
        print(np.shape(green_contours))
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
            
        
        # green_c = max(green_contours, key=cv2.contourArea)

        # Draw a bounding box around the green contour
        x2, y2, w2, h2 = cv2.boundingRect(new_contour_g)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

        # Calculate the center of the bounding box
        center = (x2 + int(w2/2), y2 + int(h2/2))

        # Put the coordinates on the bounding box
        cv2.putText(frame, str(center), (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(x1,x2,y1,y2)
    x=int((x1+x2)/2)
    y=int((y1+y2)/2)
    z=np.round(distfunc(x1,y1,x2,y2),2)
    print(z)
    print("fps:",cap.get(cv2.CAP_PROP_FPS),end=" ")
    print("position:", (x,y,z)) 

    # Display the resulting frame
    cv2.imshow("Frame", frame)

        # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()