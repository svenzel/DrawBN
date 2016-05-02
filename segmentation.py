import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg

image = mpimg.imread("flower.jpg")

class segmentHSV:
    def __init__(self, imageRGB):
        if imageRGB.max() > 1:
            imageRGB = imageRGB / 255.
        y, x, c = imageRGB.shape
        imagesize = float(y * x)
        self.imageRGB = misc.imresize(imageRGB, (500**2)/imagesize)
        self.imageHSV = colors.rgb_to_hsv(self.imageRGB)
        self.imageHeight = self.imageRGB.shape[0]
        self.imageWidth = self.imageRGB.shape[1]
        self.segmentedRGB = -1
        self.segmentedHSV = -1
    
    # get-methods
    def RGB(self):
        return self.imageRGB
        
    def HSV(self):
        return self.imageHSV
        
    def RGBSegment(self):
        return self.segmentedRGB
        
    def HSVSegment(self):
        return self.segmentedHSV
        
    def updateRGBSegment(self):
        self.segmentedRGB = colors.hsv_to_rgb(self.segmentedHSV)
    
    # creates segmentation from imageHSV
    def segment(self, segH, segS=3, segV=3):
        segmentsH = np.linspace(0, 1, segH + 1)
        segmentsS = np.linspace(0, 1, segS + 1)
        segmentsV = np.linspace(0, 1, segV + 1)
        workingImage = self.imageHSV.copy()
        
        for i in range(segH):
            # value the values between segmentsH[i]/segmentsH[i+1] will be cast to
            valCast = segmentsH[i+1]        
            mask = (self.imageHSV[:,:,0] >= segmentsH[i]) & (self.imageHSV[:,:,0] <= segmentsH[i+1])            
            workingImage[mask, 0] = valCast
        
        #workingImage[:,:,1] = 1
        workingImage[:,:,2] = 1
        
        
        self.segmentedHSV = workingImage
        self.updateRGBSegment()
        
    
    
    