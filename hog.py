import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
def gaussian(x,y,std):
    return (1/((std**2)*(2*math.pi))*np.exp(((-1*(x**2 + y**2))/2*(std**2))))
originalimg = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0133.jpeg',0)
originalimg=cv2.resize(originalimg, (64,128))
def conv(originalimg, k,size,std1):
    if(k=='averaging'):
        gaus = np.ones((size,size))/9
    elif(k=='gaussian'):
        gaus = np.zeros((3,3))
        std = std1
        mean = np.array(np.arange(-3,3))
        for i in range(-1,2):
            for j in range(-1,2):
                gaus[i+1][j+1] = gaussian(i,j,std1)
        # print('gaus',gaus)
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

    c = np.zeros((int(originalimg.shape[0]),int(originalimg.shape[1])))
    for i in range(0,int(originalimg.shape[0])):
        for j in range(0,int(originalimg.shape[1])):
            patch = originalimg[i:i+3,j:j+3]
            try:
                c[i+1][j+1] = (np.sum(np.multiply(patch,gaus)))
            except:
                pass
    return c

fx = conv(originalimg,'p1',3,1)
fy = conv(originalimg,'p2',3,1)
m = np.sqrt(fx**2 + fy**2)
plt.subplot(1,2,1)
plt.imshow(m,cmap='gray')
plt.title('magnitude')
orientation = np.arctan(fy/fx)
plt.subplot(1,2,2)
plt.hist(orientation)
plt.title('direction')
plt.show()
