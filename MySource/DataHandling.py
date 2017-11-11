'''
Created on Nov 4, 2017

@author: andreas
'''

import glob
from moviepy.editor import VideoFileClip
import pickle

class DataHandling:
    def __init__(self):
        self.carDataPathes = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/vehicles/*"
        self.nonCarDataPathes = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/non-vehicles/*"
        self.projectVideoPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/project_video.mp4"
        self.processedDataPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/Data/ProcessedData.p"
        self.features = 'features'
        self.labels = 'labels'
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


    def SavePreProcessedData(self, features, labels):
        dataPickle = {}
        dataPickle[self.features] = features 
        dataPickle[self.labels] = labels
        pickle.dump( dataPickle, open( self.processedDataPath, "wb" ) )

    def LoadPreProcessedData(self):
        dataPickle = pickle.load(open( self.processedDataPath, "rb" ) )
        features = dataPickle[self.features]
        labels = dataPickle[self.labels]
        return features, labels 