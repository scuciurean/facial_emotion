import numpy as np

def rgb2gray(rgb):
	data = (0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1440 * rgb[:,:,2])
	data = data / data.max() * 255 # Normalizes data in range 0 - 255
	return data.astype(np.uint8)

def thresh(img, threhold):
	img[img > thresh] = 255
	img[img < thresh] = 0

	return img
