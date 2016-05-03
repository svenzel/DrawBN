import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import misc


class castClrs:
    """Class to  discretize the colorchannel of the argument image.
    
    """
    def __init__(self, imageRGB):
        # arbitrary resizing    
        #y, x, c = imageRGB.shape
        #imagesize = float(y * x)
        #self.imageRGB = misc.imresize(imageRGB, (500**2)/imagesize)
        self.imageRGB = imageRGB
        self.imageHSV = colors.rgb_to_hsv(self.imageRGB)
        self.imageHeight = self.imageRGB.shape[0]
        self.imageWidth = self.imageRGB.shape[1]
        self.segmentedRGB = -1
        self.segmentedHSV = -1
    
    
    def RGB(self):
        """Returns the image in RGB format."""
        return self.imageRGB
        
    def HSV(self):
        """Returns the image in HSV format."""
        return self.imageHSV
        
    def RGBSegment(self):
        """Returns the colorspace-segmented image in RGB format."""
        return self.segmentedRGB
        
    def HSVSegment(self):
        """Returns the colorspace-segmented image in HSV format."""
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
        # exception handling mode
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
        
        int0 = np.linspace(0, 255, segC0 + 1)
        int1 = np.linspace(0, 255, segC1 + 1)
        int2 = np.linspace(0, 255, segC2 + 1)
        
        for i in range(segC0):
            # value the values between int[i]/int[i+1] will be cast to
            valCast = valueCast(int0, i, mode)       
            mask = (workingImage[:,:,0] >= int0[i]) & (workingImage[:,:,0] <= int0[i+1])            
            workingImage[mask, 0] = valCast
        for i in range(segC1):
            # value the values between int[i]/int[i+1] will be cast to
            valCast = valueCast(int1, i, mode)       
            mask = (workingImage[:,:,1] >= int1[i]) & (workingImage[:,:,1] <= int0[i+1])            
            workingImage[mask, 1] = valCast
        for i in range(segC2):
            # value the values between int[i]/int[i+1] will be cast to
            valCast = valueCast(int2, i, mode)       
            mask = (workingImage[:,:,2] >= int2[i]) & (workingImage[:,:,2] <= int0[i+1])            
            workingImage[mask, 2] = valCast       
        
        # update segmentedImages
        if (clrs == 'rgb'):
            self.segmentedRGB = workingImage
            self.updateHSVSegment()
        if (clrs == 'hsv'):
            self.segmentedHSV = workingImage
            self.updateRGBSegment()
        
        
        
    

    
    
    