import cv2
import numpy as np

img = cv2.imread('C:/Users/Nilni Kamburugamuwa/Downloads/background.png')
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
i=0
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.imwrite('img_{}.jpg'.format(i), img[y:y+h,x:x+w])
    i += 1
    
cv2.imshow('image', img)
cv2.imshow('Binary',thresh_img)
img2 = cv2.imread('img_{}.jpg')
cv2.imshow(img2)
cv2.waitKey()

