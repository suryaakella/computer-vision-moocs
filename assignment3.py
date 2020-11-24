import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from math import exp
""" harris corner detection """
def gaussian(x,y,std):
    return 1/((std)*math.sqrt(2*math.pi))*np.exp(((-1*(x**2 + y**2))/2*(std**2)))

originalimg = cv2.imread('/home/surya/Documents/computer_vision/CAP5415_Fall2012_PA3_data/Seq1/0133.jpeg',0)
originalimg = originalimg/255.0

def conv(originalimg, k,size):
    if(k=='averaging'):
        gaus = np.ones((size,size))/9
    elif(k=='gaussian'):
        gaus = np.zeros((3,3))
        # print('enter sigma')
        std = 8
        mean = np.array(np.arange(-3,3))
        for i in range(-1,2):
            for j in range(-1,2):
                gaus[i+1][j+1] = gaussian(i,j,25/255)
        # print(gaus)
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
    for i in range(0,int(originalimg.shape[0])-4):
        # print(i)
        for j in range(0,int(originalimg.shape[1])-4):
            # print(j)
            patch = originalimg[i:i+3,j:j+3]
            # print(patch.shape)
            try:
                c[i+1][j+1] = (np.sum(np.multiply(patch,gaus)))
            except:
                pass
    return c

if __name__ == "__main__":
    fx = conv(originalimg,'p1',3)
    fy = conv(originalimg,'p2',3)
    Ixx = fx**2
    # plt.imshow(Ixx,cmap='gray')
    # plt.show()
    Iyy = fy**2
    # plt.imshow(Iyy,cmap='gray')
    # plt.show()
    Ixy = fx*fy
    # plt.imshow(Ixy,cmap='gray')
    # plt.show()
    c1 = conv(Ixx,'gaussian', 3)
    c2 = conv(Iyy,'gaussian', 3)
    c3 = conv(Ixy,'gaussian', 3)
    # plt.subplot(2,2,1)
    # plt.imshow(c1,cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(c2,cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(c3,cmap='gray')
    # plt.show()
    R = np.zeros((c1.shape[0],c1.shape[1]))
    for i in range(c1.shape[0]):
        for j in range(c2.shape[1]):
            M = np.array([[c1[i][j],c3[i][j]],[c3[i][j],c2[i][j]]])
            v,w=np.linalg.eig(M)
            R[i][j] = v[0]*v[1]/(v[0]+v[1])
            if(abs(R[i][j]) < 0.8):
                R[i][j] = 0
    print(np.argmax(R[5:10][5:10]))
    # q=50
    # print(q/2)
    # for i in range(0,c1.shape[0],int(q/2)):
    #     for j in range(0,c2.shape[1],int(q/2)):
    #         patch = originalimg[i:i+q,j:j+q]
    #         try:
    #             if(patch[25,25] == np.max(patch)):
    #                 pass
    #             else:
    #                 R[i][j]=0
    #         except:
    #             pass

    plt.imshow(np.abs(R),cmap='gray')
    print(np.count_nonzero(R))
    plt.show()
    """ end of harris corner detection"""
