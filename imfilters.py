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
