import cv2
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as lng
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
    wMat = -1 * wMat

    sh = wMat.shape

    for r in range(0, sh[0]):
        val = abs(np.sum(wMat[r], dtype=np.float32))
        wMat[r, r] = val

    return wMat

def wLLAndwLU(wMat, img, labels):
    sh = img.shape

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            wX = y * sh[1] + x

            for item in labels:
                wY = item.GetIndex()

                if wX != wY:
                    wMat[wY, wX] = 0.0
                else:
                    wMat[wY, wX] = 1.0

    return wMat


def wUL(wMat, img, labels):
    sh = img.shape

    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            wY = y * sh[1] + x

            for item in labels:
                wX = item.GetIndex()

                if wY != wX:
                    wMat[wY, wX] = 0.0

    return wMat


def computeQ(wMat, img, labels):
    sh = img.shape
    q = []

    for y in range(0, sh[0] * sh[1]):
        sum = 0.0

        for item in labels:
            x = item.GetIndex()

            sum += wMat[y, x] * img[item.GetY(), item.GetX()]

        q.append(sum)

    return q


def fullWeight(img, phoParam, labParam):
    sh = img.shape

    geoWeight = wGeo(img)
    phoWeight = wPho(img)
    labels = cr_labels(img)
    labels = labels[0] + labels[1]
    labWeight = wLab(img, labels)


    for y in range(0, sh[0]):
        for x in range(0, sh[1]):
            if(x != y):
                pix1 = phoWeight[x, y]
                pix2 = phoWeight[y, x]

                if(pix1 > pix2):
                    phoWeight[x, y] = pix2
                elif(pix2 > pix1):
                    phoWeight[y, x] = pix1

    wPrim = (1.0 / (1 + phoParam)) * geoWeight + (phoParam / (1 + phoParam)) * phoWeight
    wMat = (1.0 / (1 + labParam)) * wPrim + (labParam / (1 + labParam)) * labWeight

    wMat = wLLAndwLU(wMat, img, labels)
    q = computeQ(wMat, img, labels)
    wMat = diagonal(wMat)
    wMat = wUL(wMat, img, labels)

    b = lng.cg(wMat, q)

    return b


img = cv2.imread("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCVTest\\x64\\Debug\\bacteries.png", cv2.IMREAD_GRAYSCALE)

gauss_img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True, var=0.7)  # var can be changed

cv2.imwrite("..\\images\\original.jpg", img)
plt.imsave('..\\images\\gaussian_noise.png', gauss_img, cmap=plt.cm.gray)

#Smaller image for faster processing. Good for testing.
#gauss_img = gauss_img[:60, :60]

b = fullWeight(gauss_img, 0.1, 1 /12) # v_pho and v_lab parameters

sh = gauss_img.shape
segmented = np.zeros(sh)

for i in range(0, len(b[0])):
    segmented[i // sh[1]][i % sh[1]] = b[0][i]

processedimg = cv2.convertScaleAbs(segmented)
plt.imsave('..\\images\\segmented.png',segmented, cmap=plt.cm.gray)

ret2, th2 = cv2.threshold(processedimg,0,255,cv2.THRESH_OTSU)
cv2.imwrite("..\\images\\threshold.jpg", th2)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', gauss_img)
cv2.namedWindow('segmented', cv2.WINDOW_NORMAL)
cv2.imshow('segmented', segmented)
cv2.namedWindow('th2', cv2.WINDOW_NORMAL)
cv2.imshow('th2', th2)
cv2.waitKey(0)
cv2.destroyAllWindows()