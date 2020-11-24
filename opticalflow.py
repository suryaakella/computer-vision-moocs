import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
originalimg1 = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0133.jpeg',0)
originalimg2 = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0137.jpeg',0)
originalimg3 = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0141.jpeg',0)
originalimg1 =originalimg1/255.0
originalimg2 = originalimg2/255.0
originalimg3 =originalimg3/255.0
def conv(originalimg, k):
    if(k=='fx'):
        gaus = np.array([[1,-1],[1,-1]])
    if(k=='fy'):
        gaus = np.array([[-1,-1],[1,1]])
    if(k=='ft1'):
        gaus = np.array([[-1,-1],[-1,-1]])
    if(k=='ft2'):
        gaus = np.array([[1,1],[1,1]])

    c = np.zeros((int(originalimg.shape[0]),int(originalimg.shape[1])))
    for i in range(0,int(originalimg.shape[0])):
        for j in range(0,int(originalimg.shape[1])):
            patch = originalimg[i:i+2,j:j+2]
            try:
                c[i+1][j+1] = (np.sum(np.multiply(patch,gaus)))
            except:
                pass
    return c
def f(originalimg1,originalimg2):
    fx1 = conv(originalimg1,'fx')
    fx2 = conv(originalimg2,'fx')
    fx = (fx1+fx2)/2
    fy1 = conv(originalimg1,'fy')
    fy2 = conv(originalimg2,'fy')
    fy = (fy1+fy2)/2
    ft1 = conv(originalimg1,'ft1')
    ft2 = conv(originalimg2,'ft2')
    ft = (ft1+ft2)/2
    # plt.subplot(2,2,1)
    # plt.imshow(fx)
    # plt.subplot(2,2,2)
    # plt.imshow(fy)
    # plt.subplot(2,2,3)
    # plt.imshow(ft)
    # plt.show()
    u = np.zeros((int(originalimg1.shape[0]+5),int(originalimg1.shape[1]+5)))
    v = np.zeros((int(originalimg1.shape[0]+5),int(originalimg1.shape[1]+5)))
    for i in range(0,int(originalimg1.shape[0])):
        for j in range(0,int(originalimg1.shape[1])):
            patch1 = fx[i:i+3,j:j+3]
            patch2 = fy[i:i+3,j:j+3]
            patch3 = ft[i:i+3,j:j+3]
            # print(patch1.shape)
            # print(patch2.shape)
            # print(np.matmul(patch1,patch2))
            denominator = np.sum(patch1**2)*np.sum(patch2**2) -(np.sum((patch1*patch2)))**2
            n1 =  -np.sum(patch2**2)*(np.sum(patch1*patch3)) + (np.sum(patch1*patch2))*(np.sum(patch2*patch3))
            n2 = (np.sum(patch1*patch3))*(np.sum(patch1*patch2)) - np.sum(patch1**2)*(np.sum(patch2*patch3))
            u[i+1][j+1] = n1/denominator
            v[i+1][j+1]= n2/denominator
    return u,v
            # print(n1/denominator)
    # plt.subplot(1,2,1)
    # plt.imshow(np.abs(u),cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(np.abs(v),cmap='gray')
    # plt.show()
u1,v1 = f(originalimg1,originalimg2)
u2,v2 = f(originalimg2,originalimg3)
u = u1+u2
v = v1+v2
plt.subplot(1,2,1)
plt.imshow(np.abs(u1),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(np.abs(u2),cmap='gray')
plt.show()
