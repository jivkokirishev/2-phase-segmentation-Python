class Weight:
    __fPixel = None
    __sPixel = None
    __weight = None

    def __init__(self, fPixel, sPixel, weight):
        self.__fPixel = fPixel
        self.__sPixel = sPixel
        self.__weight = weight

    def GetFPixel(self):
        return self.__fPixel

    def GetSPixel(self):
        return self.__sPixel

    def GetWeight(self):
        return self.__weight