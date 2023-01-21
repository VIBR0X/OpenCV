import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

#### importing and reading image
# img = cv.imread("D:\opencv\Asus.jpg")
# cv.imshow('A', img)
#cv.waitKey(0)

#### Using webcam
  
# while(True):
#     ret, frame = vid.read()
  
#     cv.imshow('frame', frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv.destroyAllWindows()

#### Detecting blue in video


while(1):
    # Take each frame
    _, frame = vid.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
cv.destroyAllWindows()
