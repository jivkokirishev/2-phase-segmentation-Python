import cv2
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
import skimage
import math

from src.Pixel import Pixel
from src.Weight import Weight

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
    sh = img.shape

    labels_obj = []
    labels_nonobj = []

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            pix_val = img[y, x]

            l2 = neighbours_L2(img, y, x)
            inf = neighbours_inf(img, y, x)
            sum_l2 = 0.0
            for item in l2:
                sum_l2 += (pix_val - item[0].GetVal()) ** 2

            sum_inf = 0.0
            for item in inf:
                sum_inf += (pix_val - item[0].GetVal()) ** 2

            labels_obj.append(Pixel(y, x, pix_val - 1 / 2 * sum_l2 - 1 / 4 * sum_inf, y * sh[1] + x))
            labels_nonobj.append(Pixel(y, x, 1 - pix_val - 1 / 2 * sum_l2 - 1 / 4 * sum_inf, y * sh[1] + x))

    labels_obj.sort(key=lambda obj: obj.GetVal())
    labels_obj = labels_obj[-3:]
    labels_nonobj.sort(key=lambda obj: obj.GetVal())
    labels_nonobj = labels_nonobj[-3:]

    for item in labels_obj:
        img[item.GetY(), item.GetX()] = 1

    for item in labels_nonobj:
        img[item.GetY(), item.GetX()] = 0

    return [labels_obj, labels_nonobj]


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
    sh = img.shape
    n = sh[0] * sh[1]
    wMat = sp.lil_matrix((n, n), dtype=np.float32)

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            pixels = neighbours_inf(img, y, x, 2)
            weights = []
            for pix in pixels:
                w = Weight(Pixel(y, x, img[y, x], y * sh[1] + x), pix[0], math.exp(
                    -(avg(img, y, x) - avg(img, pix[0].GetY(), pix[0].GetX())) ** 2 - np.linalg.norm(
                        np.array([pix[0].GetY(), pix[0].GetX()]) - np.array([y, x])) ** 2))
                weights.append(w)

            weights.sort(key=lambda obj: obj.GetWeight())
            sum = 0.0
            for item in weights[-4:]:
                sum += item.GetWeight()

            for item in weights[-4:]:
                wMat[item.GetFPixel().GetIndex(), item.GetSPixel().GetIndex()] = item.GetWeight() / sum

    return wMat


def wLab(img, labels):
    sh = img.shape
    n = sh[0] * sh[1]
    wMat = sp.lil_matrix((n, n), dtype=np.float32)

    labels_avg = []
    for item in labels:
        labels_avg.append(
            Pixel(item.GetY(), item.GetX(), avg(img, item.GetY(), item.GetX()), item.GetY() * sh[1] + item.GetX()))

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            avg_val = avg(img, y, x)
            weight_y = y * sh[1] + x

            weights = []
            sum = 0.0
            for item in labels_avg:
                e = math.exp(-(item.GetVal() - avg_val) ** 2)
                sum += e
                weights.append(Weight(Pixel(y, x, img[y, x]), item, e))

            for item in weights:
                wMat[weight_y, item.GetSPixel().GetIndex()] = item.GetWeight() / sum

    return wMat


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
