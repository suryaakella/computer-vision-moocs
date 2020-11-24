import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from math import exp

originalimg = cv2.imread('/home/surya/Documents/computer_vision/08.png',0)
originalimg= originalimg/255.0
o1 = originalimg

# gaussian noise

gaus = np.zeros((3,3))
def gaussian(x,y,std):
    return 1/((std)*math.sqrt(2*math.pi))*np.exp(((-1*(x**2 + y**2))/2*(std**2)))
mean = np.array(np.arange(-3,3))
for i in range(-1,2):
    for j in range(-1,2):
        gaus[i+1][j+1] = gaussian(i,j,25/255)
print(gaus)
# plt.plot(mean,gaus)
# plt.show()

# image filtering
originalimg = originalimg + np.random.normal(0,25/255,(512,512))
i=0
j=0
# filter = (np.zeros((3,3)))
# filter[0][0]= 1
# filter[1][0]= 2
# filter[2][0]= 1
# filter[0][2]= -1
# filter[1][2]= -2
# filter[2][2]= -1
filter = np.random.normal(0,25/255,(3,3))
print(filter)
plt.imshow(originalimg,cmap='gray')
plt.show()
c = np.zeros((512,512))
for i in range(0,510):
    for j in range(0,510):
        patch = originalimg[i:i+3,j:j+3]
        print(j)
        print(patch.shape)
        # filter[1][] = 1
        c[i+1][j+1] = (np.sum(np.multiply(patch,gaus)))
plt.subplot(1,2,1)
plt.imshow(originalimg,cmap ='gray')
print(originalimg[:3][25:28])
print(np.squeeze(c[:3][25:28]))
plt.subplot(1,2,2)
from sklearn.metrics import mean_squared_error
inp_psnr = -10*math.log10(mean_squared_error(originalimg,o1))
out_psnr = -10*math.log10(mean_squared_error(originalimg,c))
print(c)
print(inp_psnr)
print(out_psnr)
plt.imshow(c,cmap ='gray')
plt.title('filter krne k baad')
plt.show()

# a = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]), (1,10))
# filter = np.reshape(np.array([-1, 1]), (2,1))
# b = np.zeros((10,10)).tolist()
# a.tolist()
# filter.tolist()
# print(a[0][0:2].shape)
# print(filter.shape)
# for i in range(0, 9):
#     # print(i)
#     print(np.dot(a[0][i:i+2], filter).tolist()[0])
#     b[i+1]=(np.dot(a[0][i:i+2], filter).tolist()[0])
# print(b)

# def derivative(image):
#     return np.dot(image, [-1,1])
# d = derivative(originalimg)
# plt.imshow(d)
# plt.show()
