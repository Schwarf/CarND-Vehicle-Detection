'''
Created on Nov 5, 2017

@author: andreas
'''

import cv2
import numpy as np
from skimage.feature import hog
from collections import namedtuple
from DataHandling import DataHandling
from sklearn.preprocessing import StandardScaler
from DataAugmentation import *

HogParameters = namedtuple("HogParameters", "isOn orientationsCount pixelsPerCell cellsPerBlock visualize channel")
ColorParameters = namedtuple("ColorParameters", "isOn binCount")
SpatialParameters = namedtuple("SpatialParameters", "isOn spatialSize")

DefaultHogParameters = HogParameters(isOn = True, orientationsCount = 18, pixelsPerCell = (16,16), cellsPerBlock = (2,2), visualize = False, channel = "ALL")
DefaultColorParameters = ColorParameters(isOn = True, binCount = 64)
DefaultSpatialParameters = SpatialParameters(isOn =True, spatialSize = (8,8))


class FeatureProcessing:
    def __init__(self, hogParameters, colorParameters, spatialParameters, colorSpace = "HSV"):
        self.hogParameters = hogParameters
        self.colorParameters = colorParameters
        self.spatialParameters = spatialParameters 
        self.colorSpace = colorSpace
        self.data = DataHandling()
        self.PrintParameters()
        self.scaler = None
        
    def PrintParameters(self):
        print("Colorspace is ", self.colorSpace)
        for name, value in self.hogParameters._asdict().items():
            print("Hog parameter '"+name+"' is: ", value)
        for name, value in self.colorParameters._asdict().items():
            print("Color parameter '"+name+"' is: ", value)
        for name, value in self.spatialParameters._asdict().items():
            print("Spatial parameter '"+name+"' is: ", value)

    
    def ComputeSpatialFeatures(self, featureImage):
        features = []
        if self.spatialParameters.isOn == True:
            features = cv2.resize(featureImage, self.spatialParameters.spatialSize).ravel()
        return features
        
    
    def ComputeColorHistogramFeatures(self, featureImage):
        features = []
        if self.colorParameters.isOn == True:
            binsRange = (0,256)
            binCount = self.colorParameters.binCount

            channel0 = np.histogram(featureImage[:,:,0], bins=binCount, range = binsRange) 
            channel1 = np.histogram(featureImage[:,:,1], bins=binCount, range = binsRange)
            channel2 = np.histogram(featureImage[:,:,2], bins=binCount, range = binsRange)
        
            features = np.concatenate((channel0[0], channel1[0], channel2[0]))
        return features
    
    
    def GetHogFeatures(self,featureImage):
        features = []
        if self.hogParameters.isOn == True:
            pixelsPerCell = self.hogParameters.pixelsPerCell
            cellsPerBlock = self.hogParameters.cellsPerBlock
            orientationsCount = self.hogParameters.orientationsCount
            visualize = self.hogParameters.visualize
            featureVector = True
            if(visualize):
                features, hog_image = hog(featureImage, orientations=orientationsCount, pixels_per_cell=pixelsPerCell, cells_per_block=cellsPerBlock, transform_sqrt=True, visualise=visualize, feature_vector=featureVector)
            else:
                features = hog(featureImage, orientations=orientationsCount, pixels_per_cell=pixelsPerCell, cells_per_block=cellsPerBlock, transform_sqrt=True, visualise=visualize, feature_vector=featureVector)
            
        return features

    
    def ComputeGradientFeatures(self, featureImage):
        channel = self.hogParameters.channel
        features = []
        if(channel == "ALL"):
            hogFeatures = []
            
            for channel in range(featureImage.shape[2]):
                hogFeatures.append(self.GetHogFeatures(featureImage[:,:,channel]))
            hogFeatures = np.ravel(hogFeatures)
        else:
            hogFeatures = self.GetHogFeatures(featureImage[:,:,channel])
        
        
        return hogFeatures
         
    
    def SetColorSpace(self,image, colorSpace):
        featureImage = image
        if(colorSpace == "HSV"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif(colorSpace == "LUV"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif(colorSpace == "YUV"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif(colorSpace == "RGB"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif(colorSpace == "HLS"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif(colorSpace == "YCrCb"):
            featureImage = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        return featureImage
    
    def ComputeFeatures(self, imagesPathes, show=False):
        features = []
        
        for imagePath in imagesPathes:
            image = cv2.imread(imagePath)
            if(show):
                print(image.shape)
                cv2.imshow('title',image)
                cv2.waitKey(0)
            colorImage = self.SetColorSpace(image, self.colorSpace)
            
            spatialFeatures = self.ComputeSpatialFeatures(colorImage)
            colorFeatures = self.ComputeColorHistogramFeatures(colorImage)
            gradientFeatures = self.ComputeGradientFeatures(colorImage)
            
            features.append(np.concatenate((gradientFeatures, colorFeatures, spatialFeatures)))
        return features
    
    def ComputeFeatures2(self, imagesPathes, show=False, nonCar=False):
        features = []
        images = []
        for imagePath in imagesPathes:
            image = cv2.imread(imagePath)
            images = images + [image]
        
        if(nonCar):    
            print ("Before augmentation = ", len(images))
            augmentation = DataAugmentation()
            images = augmentation.DataAugmentation(images, 20000)
            print ("After augmentation = ", len(images))
        
        for image in images:
            if(show):
                print(image.shape)
                cv2.imshow('title',image)
                cv2.waitKey(0)
            colorImage = self.SetColorSpace(image, self.colorSpace)
            spatialFeatures = self.ComputeSpatialFeatures(colorImage)
            colorFeatures = self.ComputeColorHistogramFeatures(colorImage)
            gradientFeatures = self.ComputeGradientFeatures(colorImage)
            features.append(np.concatenate((gradientFeatures, colorFeatures, spatialFeatures)))
        return features
            
    def NormalizeAllFeatures(self, features):
        self.scaler = StandardScaler().fit(features)
        return self.scaler.transform(features)
    
    
    def Apply(self, image, scaler):
        features = []
        colorImage = self.SetColorSpace(image, self.colorSpace)
        
        spatialFeatures = self.ComputeSpatialFeatures(colorImage)
        colorFeatures = self.ComputeColorHistogramFeatures(colorImage)
        gradientFeatures = self.ComputeGradientFeatures(colorImage)
        
        features.append(np.concatenate((gradientFeatures, colorFeatures, spatialFeatures)))
        
        normalizedFeatures = scaler.transform(features)
        return normalizedFeatures
        
        
    
    
    def ComputeAllFeaturesAndLabels(self, storeData = False):
        carData = self.data.GetCarData()
        nonCarData = self.data.GetNonCarData()
        
        
#        carFeatures = self.ComputeFeatures2(carData, nonCar=False)
#        nonCarFeatures = self.ComputeFeatures2(nonCarData, nonCar=True)
        carFeatures = self.ComputeFeatures(carData)
        nonCarFeatures = self.ComputeFeatures(nonCarData)
        
        allFeatures = np.vstack((carFeatures, nonCarFeatures)).astype(np.float64)
        normalizedFeatures = self.NormalizeAllFeatures(allFeatures)
        labels = np.hstack((np.ones(len(carFeatures)), np.zeros(len(nonCarFeatures))))
        
        assert(len(labels) == len(normalizedFeatures))
        """
        index = np.random.randint(0,1000)
        imagePath = nonCarData[index]
        image = cv2.imread(imagePath)
        print("Shape:", image.shape)
        print("Label:", labels[len(nonCarData)+index])
        cv2.imshow('title',image)
        if cv2.waitKey(500000) & 0xFF == ord('q'):
            x=1
        """
        if(storeData):
            self.data.SavePreProcessedData(normalizedFeatures, labels, self.scaler)

        return normalizedFeatures, labels, self.scaler


#test = FeatureProcessing(DefaultHogParameters, DefaultColorParameters, DefaultSpatialParameters)
#test.ComputeAllFeaturesAndLabels() 