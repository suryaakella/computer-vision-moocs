import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
def gaussian(x,y,std):
    return (1/((std**2)*(2*math.pi))*np.exp(((-1*(x**2 + y**2))/2*(std**2))))
originalimg = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0133.jpeg',0)
originalimg=cv2.resize(originalimg, (160,120))
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

k = math.sqrt(2)
gaussian1 = conv(originalimg,'gaussian',3,1.6)
gaussian1 = gaussian1/np.max(gaussian1)
gaussian2 = conv(originalimg,'gaussian',3,k*1.6)
gaussian2 = gaussian2/np.max(gaussian2)
gaussian3 = conv(originalimg,'gaussian',3,k*k*1.6)
gaussian3 = gaussian3/np.max(gaussian3)
gaussian4 = conv(originalimg,'gaussian',3,k*k*k*1.6)
gaussian4 = gaussian4/np.max(gaussian4)
gaussian5 = conv(originalimg,'gaussian',3,k*k*k*k*1.6)
# plt.subplot(2,2,1)
# plt.imshow(gaussian1,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(gaussian2,cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(gaussian3,cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(gaussian4,cmap='gray')
# plt.show()
LOG1 = np.subtract(gaussian2,gaussian1)
LOG2 = np.subtract(gaussian3,gaussian2)
LOG3 = np.subtract(gaussian4,gaussian3)
LOG4 = np.subtract(gaussian5,gaussian4)
# plt.subplot(2,2,1)
# plt.imshow(LOG1,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(LOG2,cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(LOG3,cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(LOG4,cmap='gray')
# print(LOG4)
# plt.show()
c = np.zeros((int(LOG1.shape[0]),int(LOG2.shape[1])))
for i in range(0,LOG1.shape[0]):
    for j in range(0,LOG1.shape[1]):
        patch1 = LOG1[i:i+3,j:j+3]
        patch2 = LOG2[i:i+3,j:j+3]
        patch3 = LOG3[i:i+3,j:j+3]
        patch4 = LOG4[i:i+3,j:j+3]
        overallpatch = np.array([patch1,patch2,patch3])
        overallpatcha = np.array([patch4,patch2,patch3])
        try:
            if(np.max(overallpatch) == patch2[1][1] or np.min(overallpatch) == patch2[1][1]):
                c[i][j] = patch2[1][1]
            else:
                pass
            if(np.max(overallpatcha) == patch3[1][1] or np.min(overallpatcha) == patch3[1][1]):
                c[i][j] = patch3[1][1]
            else:
                pass
        except:
            pass
plt.imshow(abs(c))
plt.show()
# plt.subplot(1,2,2)
print(np.max(c))
print(np.min(c))
c = np.abs(c)
# c = c/np.max(c)
print(np.max(c))
print(np.min(c))
for i in range(0,c.shape[0]):
    for j in range(0,c.shape[1]):
        if(c[i][j] < 0.03):
            c[i][j]=0
plt.imshow(c,cmap='gray')
plt.show()
dx = conv(c, 'p1',3,1)
dxx = dx*dx
dy = conv(c, 'p2',3,1)
dyy = dy*dy
dxy = dx*dy
for i in range(dxx.shape[0]):
    for j in range(dxx.shape[1]):
        D = np.array([[dxx[i][j],dxy[i][j]],[dxy[i][j],dyy[i][j]]])
        values, vectors = np.linalg.eig(D)
        try:
            if((values[0]/values[1])>10):
                c[i][j]=0
        except:
            pass
print(c)
plt.subplot(1,2,2)
plt.imshow(c,cmap='gray')
print(np.count_nonzero(c))
plt.subplot(1,2,1)
plt.imshow(originalimg,cmap='gray')
plt.show()
keypoints = []
for i in range(0,c.shape[0]):
    for j in range(0,c.shape[1]):
        if(c[i][j]!=0):
            keypoints.append([i,j])
fx = conv(gaussian3,'p1',3,1)
fy = conv(gaussian3,'p2',3,1)
m = np.sqrt(fx**2 + fy**2)
plt.subplot(1,2,1)
plt.imshow(m,cmap='gray')
plt.title('magnitude')
orientation = np.arctan(fy/fx)
plt.subplot(1,2,2)
plt.hist(orientation)
plt.title('direction')
plt.show()
print(keypoints)
