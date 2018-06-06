import cv2
from matplotlib import pyplot as plt
import skimage
import math
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters


def neighbours_L2(mat, y, x, distance = 1):

    return 0

def neighbours_inf(mat, y, x, distance = 1):

    return 0

def avg(mat, y, x):

    return 0

def cr_labels(img):

    return 0

def wGeo(img):

    return 0

def wPho(img):

    return 0

def wLab(img, labels):

    return 0



def diagonal(wMat):

    return 0

def wLLAndwLU(wMat, img, labels):

    return 0

def wUL(wMat, img, labels):

    return 0

def computeQ(wMat, img, labels):

    return 0

def fullWeight(img, phoParam, labParam):

    return 0

img = cv2.imread("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCVTest\\x64\\Debug\\bacteries.png", cv2.IMREAD_GRAYSCALE)

gauss_img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True, var=0.7)  # var can be changed

cv2.imwrite("..\\images\\original.jpg", img)
plt.imsave('..\\images\\gaussian_noise.png', gauss_img, cmap=plt.cm.gray)
