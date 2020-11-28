import matplotlib.pyplot as plt
from imfilters import sobel
from util import rgb2gray
import logging
import getopt
import time
import sys

class log:
    def __init__(self):
        self.logger = logging.getLogger('LOG')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler('performance.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def log(self, info):
        self.logger.info(info)

class filter_test(log):
    def __init__(self, img_path):
        log.__init__(self)
        self.orig = plt.imread(img_path)
    
    def display(self):
        plt.figure(); plt.imshow(self.orig); plt.title('Original'); plt.axis('off');
    
class sobel_test(filter_test):
    def __init__(self, img_path):
        filter_test.__init__(self, img_path)
    
    def filter(self, thresh):
        self.log("[SOBEL] Filtering")
        runtime = time.time()
        img = rgb2gray(self.orig)
        self.mag, self.ang = sobel(img, thresh)
        runtime = time.time() - runtime
        self.log("[SOBEL] Time elapsed: " + str(runtime))

    def display(self):
        filter_test.display(self)
        plt.figure(); plt.imshow(self.mag,cmap="gray"); plt.title('Sobel Magnitude'); plt.axis('off');
        plt.figure(); plt.imshow(self.ang,cmap="gray"); plt.title('Sobel Angle'); plt.axis('off'); plt.show();

if __name__ == "__main__":
    display_results = False
    if (len(sys.argv) != 1):
        display_results = (str(sys.argv[1]) == "-d")

    test_img = 'data/lena.png'

    t_sobel = sobel_test(test_img)
    t_sobel.filter(255) # Just skip thresholding
    if (display_results):
       t_sobel.display()
