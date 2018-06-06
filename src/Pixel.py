class Pixel:
    __y = None
    __x = None
    __val = None
    __index = None

    def __init__(self, y, x, val, index = None):
        self.__y = y
        self.__x = x
        self.__val = val
        self.__index = index

    def GetX(self):
        return self.__x

    def GetY(self):
        return self.__y

    def GetVal(self):
        return self.__val

    def GetIndex(self):
        return self.__index
