from imfilters import sobel, non_maximum_supression
from util import rgb2gray
from cv2 import resize
import numpy as np

class hog:
    def __init__(self, input_size = (64, 64),
                       patch_size = (8, 8),
                       bin = 9,
                       norm = True):
        self.input_size = input_size
        self.patch_size = patch_size
        self.bin = bin
        self.norm = norm
        self.feature_size = (input_size[0] / patch_size[0])**2 * bin

    def summary(self):
        print("[HOG descriptor summary]:")
        print("\t Input size: " + str(self.input_size))
        print("\t Patch size: " + str(self.patch_size))
        print("\t Number of bins: " + str(self.bin))
        print("\t Feature size: " + str(self.feature_size))
        print("\t Normalized blocks: " + str(self.norm))

    def get_patch_hist(self, mag, ang):
        # Delta defines the width of the angle intervals
        delta = 180 / self.bin
        size = mag.shape[0]
        hog = np.zeros(self.bin)

        for i in range(size):
            # Reduce the angle to the first 2 quadrandts
            while(ang[i] > 180):
                ang[i] = ang[i] - 180
            # If the angle fits perfectly, store magnitude at a single index
            if(ang[i] % delta == 0):
                hog[int(ang[i] / delta)] += mag[i]
            # If the angle is greater than the last interval, split it between
            #   the last and the first one
            elif(ang[i] > (180 - delta)):
                proc = ang[i] / delta - int(ang[i] / delta)
                # TODO: Fix this if branch to split the proc correctly
                hog[0] += proc * mag[i]
                hog[size] += (1 - proc) * mag[i]
            else:
                diff = ang[i] / delta
                # Split the magnitude value between the two adjacent intervals
                proc = ang[i] / delta - int(ang[i] / delta)
                # If proc > 0.5, assign the bigger value to the next bin to the right
                hog[int(ang[i] / delta) + 1 * (proc > 0.5)] += proc * mag[i]
                hog[int(ang[i] / delta) + 1 * (proc < 0.5)] += (1 - proc) * mag[i]

        return hog

    def get_feature(self, window):
        if (window.ndim == 3):
            if (window.shape[2] == 3):
                window = rgb2gray(window)
        # Reshape input to window size
        if (window.shape != self.input_size):
            window = resize(window, self.input_size)
        
        h = np.int32(window.shape[0])
        w = np.int32(window.shape[1])
        
        hogs = np.zeros((int(h / self.patch_size[0]), int(w / self.patch_size[1]), self.bin))

        mag, ang = sobel(window)
        mag = non_maximum_supression(mag,ang)

        for x in range(0, h, self.patch_size[0]):
            for y in range(0, w, self.patch_size[1]):
                mag_patch = mag[x : x + self.patch_size[0], y : y + self.patch_size[1]]
                ang_patch = ang[x : x + self.patch_size[0], y : y + self.patch_size[1]]
                # Reshape the patches as 1-D arrays
                hogs[int(x / self.patch_size[0])][int(y / self.patch_size[1])]  =  \
                    self.get_patch_hist(mag_patch.ravel(order='C'),
                                        ang_patch.ravel(order='C'))
        return hogs.ravel(order='C')