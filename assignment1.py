import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from math import exp

""" assignment part 2 """
def gaussian(x,y,std):
    return 1/((std)*math.sqrt(2*math.pi))*np.exp(((-1*(x**2 + y**2))/2*(std**2)))

originalimg = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA1/balloonGrayNoisy.jpg',0)
originalimg=originalimg/255.0
print(originalimg.shape)
plt.subplot(2,2,1)
plt.imshow(originalimg,cmap='gray')
plt.title('original image')
def conv(originalimg, k,size):
    if(k=='averaging'):
        gaus = np.ones((size,size))/9
    elif(k=='gaussian'):
        gaus = np.zeros((3,3))
        print('enter sigma')
        std = int(input())
        mean = np.array(np.arange(-3,3))
        for i in range(-1,2):
            for j in range(-1,2):
                gaus[i+1][j+1] = gaussian(i,j,25/255)
        print(gaus)
    elif(k=='sobel1'):
        gaus = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        size = 3
    elif(k=='sobel2'):
        gaus = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        size = 3
    elif(k=='p1'):
        gaus = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        size = 3
    elif(k=='p2'):
        gaus = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        size = 3

    c = np.zeros((int(originalimg.shape[0])-3,int(originalimg.shape[1])-3))
    for i in range(0,int(originalimg.shape[0])-4):
        print(i)
        for j in range(0,int(originalimg.shape[1])-4):
            print(j)
            patch = originalimg[i:i+3,j:j+3]
            print(patch.shape)
            try:
                c[i+1][j+1] = (np.sum(np.multiply(patch,gaus)))
            except:
                pass
    return c
""" end of assignment part 1"""

""" assignment part 2 """
averaged = conv(originalimg,'averaging', 3)
plt.subplot(2,2,2)
plt.imshow(averaged,cmap = 'gray')
plt.title('average')
gaussian_smoothing = conv(originalimg,'gaussian', 3)
plt.subplot(2,2,3)
plt.imshow(gaussian_smoothing,cmap='gray')
plt.title('gaussian_smoothing')
plt.show()
""" end of assignment part 2"""

""" assignment part 3 """
originalimg = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA1/buildingGray.jpg',0)

originalimg=originalimg/255.0
plt.subplot(2,2,1)
plt.imshow(originalimg)
plt.title('original image')
fx = conv(originalimg,'sobel1',3)
plt.subplot(2,2,2)
plt.imshow(fx)
plt.title('derivative x')
fy = conv(originalimg,'sobel2',3)
plt.subplot(2,2,3)
plt.imshow(fy)
plt.title('derivative y')
magnitude = np.sqrt(fx**2 +fy**2)
plt.subplot(2,2,4)
plt.imshow(magnitude)
plt.show()
magnitude = magnitude/np.max(magnitude)
plt.subplot(1,2,1)
plt.imshow(magnitude)
for i in range(0,magnitude.shape[0]):
    for j in range(0,magnitude.shape[1]):
        if(magnitude[i][j] < 0.2): # threshold value = 0.2
            magnitude[i][j]=0
plt.subplot(1,2,2)
plt.imshow(magnitude)
plt.title('after thresholding')
plt.show()

""" end of assignment part 3"""
