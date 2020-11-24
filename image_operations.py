import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from math import exp
originalimg = cv2.imread('/home/surya/Documents/computer_vision/08.png',0)
""" image translation """
# copyimg = np.zeros((1000,1000))
# dx = 0
# dy = 100
# translate = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
#
# for i in range(0,512):
#     for j in range(0,512):
#         abc = np.matmul(translate, np.array([i, j, 1]))
#         copyimg[abc[0]][abc[1]] = originalimg[i][j]
# plt.imshow(copyimg,cmap = 'gray')
# plt.show()
""" image scaling """
# s1 = 0.5
# s2 = 0.5
# copyimg = np.zeros((256,256))
#
# scale = np.array([[s1,0],[0,s2]])
# for i in range(0,512):
#     for j in range(0,512):
#         abc = np.matmul(scale, np.array([i, j]))
#         copyimg[int(abc[0])][int(abc[1])] = originalimg[i][j]
# plt.imshow(copyimg,cmap = 'gray')
# plt.show()
# plt.imshow(originalimg,cmap='gray')
# plt.show()

""" image rotation """
# x = math.pi/4
# rotate = np.array([[np.cos(x), -np.sin(x)],[np.sin(x), np.cos(x)]])
# rotated_img = np.zeros((1000,1000))
# for i in range(0,512):
#     print(i)
#     for j in range(0,512):
#          output_index = np.matmul(rotate, np.array([i, j]))
#          rotated_img[int(output_index[0])][int(output_index[1])] = originalimg[i][j]
# plt.imshow(rotated_img)
# plt.show()
