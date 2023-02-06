# press "q" to close the window

from math import dist
import cv2 as cv
from numpy.core.numeric import tensordot
import numpy as np
import time
from pyzbar.pyzbar import decode as qr_decode

def decoder(image):
    gray_img = cv.cvtColor(image,0)
    a=qr_decode(gray_img)
    if len(a)>0:
        qr=a[0]
    # qr = qr_decode(gray_img)[0]

        qrCodeData = qr.data.decode("utf-8")
        return qrCodeData

cv.namedWindow("video")
cap=cv.VideoCapture(2)
# cap = cv.VideoCapture(2, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)



while(cap.isOpened()):
    ret, frame=cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    print(decoder(frame))
    cv.imshow("video",frame)
    if(cv.waitKey(1)==ord('q')):
        # print("frame size:",outputFrame.shape[:2])        #size of your video in pixels
        break
cap.release()
# out.release()
cv.destroyAllWindows()