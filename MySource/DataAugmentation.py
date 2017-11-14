'''
Created on Nov 13, 2017

@author: andre
'''
import numpy as np
import cv2

class DataAugmentation:

    def __init__(self):
        pass

    def ApplyImageRotation(self, image, angle):
        imageCenterWidth =image.shape[0]//2
        imageCenterHeight =image.shape[1]//2
        rotationMatrix = cv2.getRotationMatrix2D((imageCenterWidth, imageCenterHeight),angle,1)
        return cv2.warpAffine(image,rotationMatrix,(image.shape[0],image.shape[1]))   
    
    def ApplyImageTranslation(self,image, translation):
        translationMatrix = np.float32([[1,0,translation[0]],[0,1,translation[1]]])
        return cv2.warpAffine(image,translationMatrix,(image.shape[0],image.shape[1]))
        
    def ApplyImageRescaling(self,image, scalingFactor):
        imageWidth = image.shape[0]
        imageHeight = image.shape[1]
        newImage = cv2.resize(image,(0,0), fx=scalingFactor, fy=scalingFactor)
        newImageWidth = newImage.shape[0]
        diffWidth = imageWidth-newImageWidth
        # No scaling applied
        if(diffWidth ==0):
            newImage = image
        #New image is smaller than original, take black image and copy new image into the black one
        elif(diffWidth > 0):
            blankImage = np.zeros((imageWidth,imageHeight,3), np.uint8)
            maxOffset = diffWidth
            offsetWidth = np.random.randint(0,maxOffset)
            offsetHeight = np.random.randint(0,maxOffset)
            blankImage[offsetWidth:offsetWidth+newImageWidth, offsetHeight:offsetHeight+newImageWidth] = newImage
            newImage = blankImage
        #New image is larger than original, take a random part
        else:
            maxOffset = -diffWidth
            offsetWidth = np.random.randint(0,maxOffset)
            offsetHeight = np.random.randint(0,maxOffset)
            newImage = newImage[offsetWidth:offsetWidth+imageWidth, offsetHeight:offsetHeight+imageWidth]
        
        assert(newImage.shape == image.shape)
        return newImage
        
        
    def TransformImage(self,image):
        angle = np.random.uniform(-15,15)
        translation = np.random.randint(-4,4,2)
        scalingFactor = np.random.uniform(0.8,1.2)
    
        rotatedImage = self.ApplyImageRotation(image, angle)
        translatedImage = self.ApplyImageTranslation(rotatedImage, translation)
        scaledImage = self.ApplyImageRescaling(translatedImage, scalingFactor)
        return scaledImage
          
        #cv2.imshow('frame', newImage)
        #if cv2.waitKey(1000) & 0xFF == ord('q'):
        #    exit()
        #cv2.imshow('frame', scaledImage)
        #if cv2.waitKey(1000) & 0xFF == ord('q'):
        #    exit()
    
    
    
    def GenerateNewData(self,numberOfDataSets, originalImages):
        generatedImageSamples = np.random.randint(len(originalImages), size = numberOfDataSets)
        
        for index in generatedImageSamples:
            originalImages = originalImages + [self.TransformImage(originalImages[index])]    
        return originalImages 
    
    
    def DataAugmentation(self, features, numberOfInstancesToGenerate):
        
        features = self.GenerateNewData(numberOfInstancesToGenerate, features)
        return features
