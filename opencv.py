import cv2 as cv
import cvzone
from cvzone.ColorModule import ColorFinder 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as transform




#### importing and reading image

# img = cv.imread("Asus.jpg")
# cv.imshow('A', img)
# cv.waitKey(0)



#### Using webcam
#vid = cv.VideoCapture(0)
# while(True):
#     ret, frame = vid.read()
  
#     cv.imshow('frame', frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv.destroyAllWindows()




#### image transform

# img1 = cv.imread("t.jpg")
# cv.imshow('img', img1)
# grey = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# #cv.imshow('grey', grey)
# blur = cv.GaussianBlur(img1, (5,5), cv.BORDER_DEFAULT)
# #cv.imshow('blur',blur)
# canny = cv.Canny(img1, 100, 200)
# cv.imshow('canny', canny)
# cv.waitKey(0)





#### Detecting Shapes

# img = cv.imread('shapes.png')
  
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
# _, threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
  

# contours, _ = cv.findContours(
#     threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  
# i = 0
  
# # list for storing names of shapes
# for contour in contours:
  
  
#     if i == 0:
#         i = 1
#         continue
    
#     approx = cv.approxPolyDP(
#         contour, 0.01 * cv.arcLength(contour, True), True)
      
    
#     cv.drawContours(img, [contour], 0, (0, 0, 255), 5)
  
#     M = cv.moments(contour)
#     if M['m00'] != 0.0:
#         x = int(M['m10']/M['m00'])
#         y = int(M['m01']/M['m00'])
  
#     # putting shape name at center of each shape
#     if len(approx) == 3:
#         cv.putText(img, 'Triangle', (x, y),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
#     elif len(approx) == 4:
#         cv.putText(img, 'Quadrilateral', (x, y),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
#     elif len(approx) == 5:
#         cv.putText(img, 'Pentagon', (x, y),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
#     elif len(approx) == 6:
#         cv.putText(img, 'Hexagon', (x, y),
#           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
#     else:
#         cv.putText(img, 'circle', (x, y),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
# cv.imshow('shapes', img)
  
# cv.waitKey(0)
# cv.destroyAllWindows()





#### Detecting Ball in the video
# vid = cv.VideoCapture('messi.mp4')
# prevc = None
# dist = lambda x1,y1,x2,y2: ((x1-x2)**2+(y1-y2)**2)**0.5
# while(True):
#     ret, frame = vid.read()
#     grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     blur = cv.GaussianBlur(grey, (17,17), cv.BORDER_DEFAULT)
#     circle = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.3, 1000, param1=100, param2=40)

#     if circle is not None:
#         circle = np.uint16(np.around(circle))
#         sel = None
#         for i in circle[0,:]:
#             if sel is None: sel = i
#             if prevc is not None:
#                 if dist(sel[0],sel[1],prevc[0],prevc[1]) <= dist(i[0],i[1],prevc[0],prevc[1]):
#                     sel = i
#         cv.circle(frame, (sel[0],sel[1]), 2, (100,0,100), 2 )
#         cv.circle(frame, (sel[0],sel[1]), sel[2], (255,0,0), 4 )
#         prevc = sel
  
#     cv.imshow('frame', frame)

#     if cv.waitKey(600) & 0xFF == ord('q'):
#         break

# vid.release()
# cv.destroyAllWindows()





#### Detecting blue in video

# while(1):
#     # Take each frame
#     _, frame = vid.read()
#     # Convert BGR to HSV
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     # define range of blue color in HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])
#     # Threshold the HSV image to get only blue colors
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#     # Bitwise-AND mask and original image
#     res = cv.bitwise_and(frame,frame, mask= mask)
#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#       break
# cv.destroyAllWindows()



#### Detecting Ball in the video using color

vid = cv.VideoCapture('video.mp4')
colorf = ColorFinder(False)

hsvVal = {'hmin': 23, 'smin': 72, 'vmin': 204, 'hmax': 47, 'smax': 255, 'vmax': 255}

while(True):
    ret, frame = vid.read()

    # img = cv.imread('ball.png')
    # cv.imshow('img', img)

    # imgcol,mask = colorf.update(img, hsvVal)
    # cv.imshow('imgcol', imgcol)

    # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # blur = cv.GaussianBlur(grey, (17,17), cv.BORDER_DEFAULT)

    imgcol,mask = colorf.update(frame, hsvVal)  

    imgcontours, contours = cvzone.findContours(frame,mask, minArea=200)
    cv.imshow('frame', imgcontours)
    #cv.imshow('frame2', frame)

    if cv.waitKey(60) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()

