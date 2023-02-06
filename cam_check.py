# Python program to check
# whether the camera is opened
# or not


import numpy as np
import cv2


cap = cv2.VideoCapture(2)
while(cap.isOpened()):
    while True:
        ret, img = cap.read()
        cv2.imshow('img', img)
        print(cap.get(cv2.CAP_PROP_FPS))
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
			
    cap.release()
    cv2.destroyAllWindows()
else:
	print("Alert ! Camera disconnected")
