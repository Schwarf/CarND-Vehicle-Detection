'''
Created on Nov 4, 2017

@author: andreas
'''

import glob
from moviepy.editor import VideoFileClip
import pickle
import cv2

class DataHandling:
    def __init__(self):
        self.carDataPathes = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/vehicles/*"
        self.nonCarDataPathes = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/non-vehicles/*"
        self.projectVideoPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/project_video.mp4"
        self.projectVideoPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/test_video.mp4"
        self.processedDataPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/ProcessedData.p"
        self.testImagesPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/test_images/*.jpg"
        self.features = 'features'
        self.labels = 'labels'
        self.scaler = 'scaler'
        self.totalDataCount = None
        
        
    def GetFiles(self, path):
        directories = [dirs for dirs in glob.glob(path)]
        files =[]
        for dir in directories:
            string = dir+"/*.png"
            string = string.replace("\\", "/")
            currentFiles = [files for files in glob.glob(string)]
            files = files + currentFiles
        return files
    
    def GetCarData(self):
        carData = self.GetFiles(self.carDataPathes)
        #print ("Number of vehicle images: ",len(carData))
        return carData
    
    def GetNonCarData(self):
        nonCarData = self.GetFiles(self.nonCarDataPathes)
        #print ("Number of non-vehicle images: ", len(nonCarData))
        return nonCarData

    def LoadProjectVideo(self):
        return VideoFileClip(self.projectVideoPath)

    def LoadProjectTestVideo(self):
        return VideoFileClip(self.projectVideoPath)

    def LoadTestImages(self):
        jpgFiles = [file for file in glob.glob(self.testImagesPath)]
        images = []
        for filePath in jpgFiles:
            image = cv2.imread(filePath)
            images.append(image)
        return images


    def SavePreProcessedData(self, features, labels, scaler):
        dataPickle = {}
        dataPickle[self.features] = features 
        dataPickle[self.labels] = labels
        dataPickle[self.scaler] = scaler
        pickle.dump( dataPickle, open( self.processedDataPath, "wb" ) )

    def LoadPreProcessedData(self):
        dataPickle = pickle.load(open( self.processedDataPath, "rb" ) )
        features = dataPickle[self.features]
        labels = dataPickle[self.labels]
        scaler = dataPickle[self.scaler]
        return features, labels, scaler