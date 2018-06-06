import cv2
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
import skimage
import math

from src.Pixel import Pixel

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters


def neighbours_L2(mat, y, x, distance=1):
    sh = mat.shape
    out = []
    for d in range(1, distance + 1):
        if x + d < sh[1]:
            pix = Pixel(y, x + d, mat[y, x + d], y * sh[1] + x + d)
            out.append([pix, d])

        if x - d >= 0:
            pix = Pixel(y, x - d, mat[y, x - d], y * sh[1] + x - d)
            out.append([pix, d])

        if y + d < sh[0]:
            pix = Pixel(y + d, x, mat[y + d, x], (y + d) * sh[1] + x)
            out.append([pix, d])

        if y - d >= 0:
            pix = Pixel(y - d, x, mat[y - d, x], (y - d) * sh[1] + x)
            out.append([pix, d])
    return out


def neighbours_inf(mat, y, x, distance=1):
    sh = mat.shape
    out = []

    for i in range(y - distance, y + distance + 1):
        for j in range(x - distance, x + distance + 1):

            if 0 <= i < sh[0] and 0 <= j < sh[1] and not (i == y and j == x):
                pix = Pixel(i, j, mat[i, j], i * sh[1] + j)
                d = max([abs(x - j), abs(y - i)])
                out.append([pix, d])
    return out


def avg(mat, y, x):
    weight = mat[y, x] * 4
    pixels = neighbours_L2(mat, y, x)
    count = len(pixels) + 4
    for pix in pixels:
        weight += pix[0].GetVal()

    return weight / count


def cr_labels(img):
    return 0


def wGeo(img):
    sh = img.shape
    n = sh[0] * sh[1]
    wMat = sp.lil_matrix((n, n), dtype=np.float32)

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            pixels = neighbours_L2(img, y, x)
            for pix in pixels:
                wMat[y * sh[1] + x, pix[0].GetIndex()] = 1 / 4

    return wMat


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
