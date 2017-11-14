'''
Created on Nov 6, 2017

@author: andre
'''
from collections import namedtuple
from Classification import *
from DataHandling import *
from FeatureProcessing import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

SlidingWindowParameters = namedtuple("SlidingWindowParameters", "xStart xStop yStart yStop xOverlap yOverlap xWindowSize yWindowSize")

#DefaultSlidingWindowParameters =  

class DetectingVehicles:
    def __init__(self):
        self.parameters = None
        self.data = DataHandling()
        self.Initialization()
        self.heatMap = None
        self.vehicleMarkings = []
        self.heatMapList = []
        
    
    def Initialization(self):
        print("Initialization: Features")
        myColorSpace = "LUV"
        myHogParameters = HogParameters(isOn = True, orientationsCount = 16, pixelsPerCell = (16,16), cellsPerBlock = (4,4), visualize = False, channel = 'ALL')
        myColorParameters = ColorParameters(isOn = True, binCount = 128)
        mySpatialParameters = SpatialParameters(isOn =True, spatialSize = (24,24))
        #myHogParameters = HogParameters(isOn = True, orientationsCount = 72, pixelsPerCell = (16,16), cellsPerBlock = (4,4), visualize = False, channel = 'ALL')
        #myColorParameters = ColorParameters(isOn = True, binCount = 128)
        #mySpatialParameters = SpatialParameters(isOn =True, spatialSize = (8,8))
        classifierType = "LinearSVC"
        self.classifier = Classifier(classifierType)
        self.classifier.SetFeatureProcessingParameters(myHogParameters, myColorParameters, mySpatialParameters, myColorSpace)
        self.classifier.SetTrainingAndTestData(False)
        print("Initialization: Training the classifier ... ")
        self.classifier.TrainClassifier()
        print("... finished")

    
    def SetSlidingWindowParameters(self, image):
        height = image.shape[0] 
        width = image.shape[1]
        yFactor = 400./720.
        ySizeFactor = 260./720.
        yStart =  np.int(yFactor*height)
        yStop = yStart+np.int(ySizeFactor*height)
        print ("Start/stop  ", yStart, yStop)
        self.parameters = SlidingWindowParameters(xStart = 0, xStop = width, yStart = yStart, yStop = yStop, xOverlap = 0.5, yOverlap = 0.5, xWindowSize = 64, yWindowSize =64 )
    
         
    ## We use a fixed Window search size and rescale the original image
    def SearchForVehicles(self, image, windowsList, scale):
        resultWindows = []
        countVehicle = 0
        countNonVehicle = 0
        rectangles = []
        for window in windowsList:
            #resizedImage = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            imageWindow = image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            #cv2.imshow("frame", resizedImage)
            #cv2.imshow("frame", image)
            #if cv2.waitKey(5000) & 0xFF == ord('q'):
            #    continue
            predictedLabel = self.classifier.UseClassifier(imageWindow)
            if predictedLabel == 1:
                leftTop = (np.int(window[0][0]/scale), np.int(window[0][1]/scale))
                rightBottom = (np.int(window[1][0]/scale), np.int(window[1][1]/scale))
                rectangle = [leftTop, rightBottom]
                rectangles.append(rectangle)
                countVehicle +=1
            else:
                countNonVehicle +=1
        print ("Number of identified vehicles = ", countVehicle)
        return rectangles
    
    def AddToHeatMap(self, vehicleCoordinates):
        self.heatMap[vehicleCoordinates[0][1]:vehicleCoordinates[1][1], vehicleCoordinates[0][0]:vehicleCoordinates[1][0]] +=1
        
    
    def CalculateHeatMap(self):
        self.heatMapList.append(self.heatMap)
        if(len(self.heatMapList) < 2):
            return
        if(len(self.heatMapList) > 3):
            del self.heatMapList[0]
        self.heatMap = sum(self.heatMapList)
        
          
        
        
    def FinalizeHeatMap(self, video):
        threshold = 2
        if(video):
            self.CalculateHeatMap()
            threshold =7
        self.heatMap[self.heatMap <= threshold] = 0
        vehicleMarkings = label(self.heatMap)
        for vehicle in range(1, vehicleMarkings[1]+1):
            vehicleHeat = (vehicle == vehicleMarkings[0]).nonzero()
            
            yVehicle = np.array(vehicleHeat[0])
            xVehicle = np.array(vehicleHeat[1])
            rectangle = [(np.min(xVehicle), np.min(yVehicle)), (np.max(xVehicle), np.max(yVehicle))]
            self.vehicleMarkings.append(rectangle)
            
    
    def ShowImageAsPlot(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        
    def ProcessVideo(self):
        videoClip = self.data.LoadProjectVideo()
        outputFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/outputVideo3.mp4"
        #output = videoClip.subclip(15,30).fl_image(lambda x:  self.ProcessImage(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))) 
        output = videoClip.fl_image(lambda x:  self.ProcessImage(cv2.cvtColor(x, cv2.COLOR_RGB2BGR)))
        output.write_videofile(outputFile, audio=False)

    def ProcessTestVideo(self):
        videoClip = self.data.LoadProjectTestVideo()
        outputFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project5/CarND-Vehicle-Detection/outputTestVideo.mp4"
        #output = videoClip.subclip(15,30).fl_image(lambda x:  self.ProcessImage(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))) 
        output = videoClip.fl_image(lambda x:  self.ProcessImage(cv2.cvtColor(x, cv2.COLOR_RGB2BGR)))
        output.write_videofile(outputFile, audio=False)


    
    def ProcessImage(self, image):
        scaleList = [0.75, 1.0, 1.25]
        self.vehicleMarkings = []
        imageCopy = np.copy(image)
        self.heatMap = np.zeros_like(image[:,:,0])
        vehicles =[]
        for scale in scaleList:
            imageShape = image.shape
            resizedImage = cv2.resize(image, (np.int(imageShape[1]*scale), np.int(imageShape[0]*scale)))
            self.SetSlidingWindowParameters(resizedImage)
            windows = self.GetSlidingWindows(resizedImage)
            foundVehicles = self.SearchForVehicles(resizedImage, windows, scale)
            vehicles = vehicles + foundVehicles
        for vehicle in vehicles:
            self.AddToHeatMap(vehicle)
        self.FinalizeHeatMap(video=True)
        #print("Markings " , len(self.vehicleMarkings))
        for vehicle in self.vehicleMarkings:
            imageCopy = cv2.rectangle(image,vehicle[0],vehicle[1],(0,0,255),6)
        #self.ShowImageAsPlot(imageCopy)
        imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)
        
        return imageCopy
        
    
    def ProcessTestImages(self):
        images = self.data.LoadTestImages()
        scaleList = [0.75, 1.0, 1.25]
        for image in images:
            imageCopy = np.copy(image)
            self.heatMap = np.zeros_like(image[:,:,0])
            vehicles =[]
            for scale in scaleList:
                imageShape = image.shape
                resizedImage = cv2.resize(image, (np.int(imageShape[1]*scale), np.int(imageShape[0]*scale)))
                self.SetSlidingWindowParameters(resizedImage)
                windows = self.GetSlidingWindows(resizedImage)
                foundVehicles = self.SearchForVehicles(resizedImage, windows, scale)
                vehicles = vehicles + foundVehicles
            for vehicle in vehicles:
                self.AddToHeatMap(vehicle)
            self.FinalizeHeatMap(video=False)
            #print("Markings " , len(self.vehicleMarkings))
            for vehicle in self.vehicleMarkings:
                imageCopy = cv2.rectangle(image,vehicle[0],vehicle[1],(0,0,255),6)
            #cv2.imshow("frame", imageCopy)
            self.ShowImageAsPlot(imageCopy)
            self.vehicleMarkings = []
#            if cv2.waitKey(50000) & 0xFF == ord('q'):
#                continue

        
        
    def GetSlidingWindows(self,image):
    # If x and/or y start/stop positions not defined, set to image size
        #self.CheckParameters(image)

        scanningXSpan = self.parameters.xStop - self.parameters.xStart   
        scanningYSpan = self.parameters.yStop - self.parameters.yStart
        
        numberOfXPixelsPerStep = np.int(self.parameters.xWindowSize*(1.0 - self.parameters.xOverlap) )
        numberOfYPixelsPerStep = np.int(self.parameters.yWindowSize*(1.0 - self.parameters.yOverlap) )
        

        bufferX = np.int(self.parameters.xWindowSize*(self.parameters.xOverlap))
        bufferY = np.int(self.parameters.yWindowSize*(self.parameters.yOverlap))

        numberOfXWindows = np.int((scanningXSpan - bufferX)/numberOfXPixelsPerStep)
        numberOfYWindows = np.int((scanningYSpan - bufferY)/numberOfYPixelsPerStep)
        print("XWindows ", numberOfXWindows)
        print("YWindows", numberOfYWindows)
        windowList = []
        for yIndex in range(numberOfYWindows):
            for xIndex in range(numberOfXWindows):
                # Calculate window position
                startX = xIndex*numberOfXPixelsPerStep + self.parameters.xStart  
                endX = startX + self.parameters.xWindowSize
                startY = yIndex*numberOfYPixelsPerStep + self.parameters.yStart  
                endY = startY + self.parameters.yWindowSize
                # Append window position to list
                windowList.append(((startX, startY), (endX, endY)))
        # Return the list of windows
        return windowList


test = DetectingVehicles()
#test.ProcessTestImages()
test.ProcessTestVideo()