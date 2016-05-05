import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import ndimage
from PIL import Image
import time


class castClrs:
    """Class to  discretize the colorchannel of the argument image.
    
    """
    def __init__(self, imageRGB):
        self.imageRGB = imageRGB.copy()
        self.imageHSV = colors.rgb_to_hsv(self.imageRGB)
        self.imageHeight = self.imageRGB.shape[0]
        self.imageWidth = self.imageRGB.shape[1]
        self.segmentedRGB = None
        self.segmentedHSV = None
        self.computationTime = None
    
    
    def RGB(self):
        """Returns the image in RGB format."""
        return self.imageRGB
        
    def HSV(self):
        """Returns the image in HSV format."""
        return self.imageHSV
        
    def RGBSegment(self):
        """Returns the colorspace-segmented image in RGB format."""
        if self.segmentedRGB is None:
            print('There is no segmented image in RGB format')
            return 0
        else:
            return self.segmentedRGB
        
    def HSVSegment(self):
        """Returns the colorspace-segmented image in HSV format."""
        if self.segmentedHSV is None:
            print('There is no segmented image in RGB format')
            return 0
        else:
            return self.segmentedHSV
        
    def updateRGBSegment(self):
        """Takes HSV-segmented image and converts it to RGB to call from RGBSegment()"""
        self.segmentedRGB = colors.hsv_to_rgb(self.segmentedHSV)
        
    def updateHSVSegment(self):
        """Takes RGB-segmented image and converts it to HSV to call from HSVSegment()"""
        self.segmentedHSV = colors.rgb_to_hsv(self.segmentedRGB)
        
    
    def hist(self, binNumber=360):
        """Shows a histogram of hue channel with a default of 360 bins"""
        n, bins, patches = plt.hist(self.HSV()[:,:,0].flatten(), binNumber)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of hue channel')
        plt.show()
        
    def showRGB(self):
        """Shows a matplotlib image of the segmented RGB-image"""
        plt.imshow(self.RGBSegment())
        plt.show()
        
    def showHSV(self):
        """Shows a matplotlib image of the segmented HSV-image"""
        plt.imshow(self.HSVSegment())
        plt.show()
    
    
    def segment(self, segC0=8, segC1=8, segC2=8, clrs='rgb', mode='ceil'):
        """Does the colorspace segmentation.
        -segC0, segC1 and segC2 are the numbers of segments wanted for the respective colorchannels
        -clrs is the colorspace in which the segmentation should take place and can be 'rgb' or 'hsv'
        -mode is the way colorvalues will be cast and can be 'floor', 'ceil' or 'mean'"""
        timeStart = time.clock()
        
        # define modes
        modes = ['floor', 'ceil', 'mean']        
        # set working image
        if (clrs == 'rgb'):
            workingImage = self.imageRGB.copy()
        elif (clrs == 'hsv'):
            workingImage = self.imageHSV.copy()
        else:
            print('Colorspace "'+str(clrs)+'" not found')
            return 0
        # exception handling: mode
        if not (mode in modes):
            print('Mode "'+str(mode)+'" not found')
            return 0
            
        def valueCast(array, index, mode):
            if (mode == 'floor'):
                return int(array[index])
            elif (mode == 'ceil'):
                return int(array[index+1])
            elif (mode == 'mean'):
                return int((array[index]+array[index+1])/2.)
            else:
                return 0
            
        # needed for color comparison
        referenceImage = workingImage.copy()
        
        int0 = np.linspace(0, 255, segC0 + 1)
        int1 = np.linspace(0, 255, segC1 + 1)
        int2 = np.linspace(0, 255, segC2 + 1)
        
        for i in range(segC0):
            # value the values between int[i]/int[i+1] will be cast to
            valCast = valueCast(int0, i, mode)       
            mask = (referenceImage[:,:,0] >= int0[i]) & (referenceImage[:,:,0] <= int0[i+1])            
            workingImage[mask, 0] = valCast
        for i in range(segC1):
            valCast = valueCast(int1, i, mode)       
            mask = (referenceImage[:,:,1] >= int1[i]) & (referenceImage[:,:,1] <= int1[i+1])            
            workingImage[mask, 1] = valCast
        for i in range(segC2):
            valCast = valueCast(int2, i, mode)       
            mask = (referenceImage[:,:,2] >= int2[i]) & (referenceImage[:,:,2] <= int2[i+1])            
            workingImage[mask, 2] = valCast       
        
        # update segmentedImages
        if (clrs == 'rgb'):
            self.segmentedRGB = workingImage
            self.updateHSVSegment()
        if (clrs == 'hsv'):
            self.segmentedHSV = workingImage
            self.updateRGBSegment()
        
        self.computationTime = time.clock() - timeStart
            
    def showInfo(self):
        """Shows computation info on castClrs.segment()"""
        if self.computationTime is None:
            print('castClrs.segment() has not yet taken place')
        else:
            print('Computation time for colorspace segmentation: ' + str(self.computationTime) + 's\n')
        
        
        
class extractConnComp():
    """This class aims to extract connected components from a colorspace-segmented image.
    Connected components in this context are pixels connected spacially as well as in colorspace.
    The connected components extracted will be used for edge detection and for obtaining better colors for each individual segment by comparison with the original image.
    
    """
    def __init__(self, image):
        self.image = image.copy()
        self.connComp = None
        self.colorNumber = None
        self.connCompNumber = None
        self.computationTime = None
    

    def extract(self):
        """Computes connected component picture"""
        timeStart = time.clock()
        
        # create PIL Image Object
        pilImg = Image.fromarray(np.uint8(self.image))
        # create numpy array containing all color tuples occuring in the image
        colorArray = np.array(pilImg.getcolors(), dtype='object')[:,1]
        
        # connCompClrs is a connected component image where pixels are
        # connected in colorspace, but not in space
        self.connComp = np.zeros((self.image.shape[0], self.image.shape[1]))
        connCompCount = 1   # offset by 1 because 0 is recognized as background
        
        # each iteration will first find pixels of same color and then build
        # spacially connected components into the image connComp
        for i in range(colorArray.size):
            connCompClrs = np.uint8(((self.image[:,:,0] == colorArray[i][0]) &
                (self.image[:,:,1] == colorArray[i][1]) &
                (self.image[:,:,2] == colorArray[i][2])))
            connCompTemp, deltaConnCompCount = ndimage.label(connCompClrs)
            self.connComp += connCompTemp+connCompCount-np.uint8(connCompTemp==0)
            connCompCount += deltaConnCompCount
            
        self.colorNumber = colorArray.size
        self.connCompNumber = np.unique(self.connComp).size
        self.computationTime = (time.clock() - timeStart)
        
    def showConnComp(self):
        """Shows self.connComp with matplotlib"""
        if self.connComp is None:
            self.extract()
        plt.imshow(self.connComp);
        plt.show()
        
    def getConnComp(self):
        """Returns self.connComp"""
        if self.connComp is None:
            self.extract()
        return self.connComp
        
    def showInfo(self):
        """Shows computation info on self.extract()"""
        if self.computationTime is None:
            print('extractConnComp.extract() has not yet taken place')
        else:
            print('Computation time for connected component segmentation: ' + str(self.computationTime) + 's')
            print('Number of different colors from colorspace segmentation: ' + str(self.colorNumber))
            print('Number of connected components: ' + str(np.unique(self.connComp).size) + '\n')
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        