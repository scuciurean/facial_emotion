from util import rgb2gray
from scipy import signal
import numpy as np
import math as m

"""
    Sobel:
        Gx = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
        Gy = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])

    Sobel-Feldman:
        Gx = np.array([[-3,  0,  3], [-10,  0, 10], [-3,  0,  3]])
        Gy = np.array([[-3,-10, -3], [  0,  0,  0], [ 3, 10,  3]])

    Scharr:
        Gx = np.array([[-47,   0,  47], [-162,   0, 162], [-47,   0, 47]])
        Gy = np.array([[-47,-162, -47], [   0,   0,   0], [ 47, 162, 47]])
"""

def sobel(img):
    Gx = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
    Gy = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
    dx = signal.convolve2d(img, Gx, boundary='symm', mode='same')
    dy = signal.convolve2d(img, Gy, boundary='symm', mode='same')
    mag = np.zeros(dx.shape)
    ang = np.zeros(dy.shape)
    for x in range(0, mag.shape[0]):
        for y in range(0, mag.shape[1]):
            mag[x, y] = m.sqrt(dx[x][y]**2 + dy[x][y]**2)
            ang[x, y] = m.atan2(dy[x][y] , dx[x][y]) * 180 / m.pi * (dx[x][y] > 0)

    return mag, ang

def non_maximum_supression(img, angle):
    nms = np.zeros(img.shape)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            q = 255
            r = 255

            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                nms[i,j] = img[i,j]
            else:
                nms[i,j] = 0
    return nms
