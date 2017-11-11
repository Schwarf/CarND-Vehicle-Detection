'''
Created on Nov 6, 2017

@author: andre
'''

SlidingWindowParameters = namedtuple("SlidingWindowParameters", "xStart xStop yStart yStop xOverlap yOverlap xWindowSize yWindowSize")

DefaultSlidingWIndowParameters = SlidingWindowParameters(xStart = None, xStop = None, yStart = None, yStop = None, xOverlap = 0.5, yOverlap = 0.5, xWindowSize = 64, yWindowSize =64 ) 

class SlidingWindow:
    def __init__(self, parameters):
        self.parameters = parameters
    
    def CheckParameters(self, image):
        if self.parameters.xStart is None:
            self.parameters.xStart = 0
        if self.parameters.xStop is None:
            self.parameters.xStop = images.shape[1]
        if self.parameters.yStart is None:
            self.parameters.yStart = 0
        if self.parameters.yStop is None:
            self.parameters.yStop = images.shape[0]
        
        
    def slidingWindows(image):
    # If x and/or y start/stop positions not defined, set to image size
        self.CheckParameters(image)

        scanningXSpan = self.parameters.xStop - self.parameters.xStart   
        scanningYSpan = self.parameters.yStop - self.parameters.yStart
        
        numberOfXPixelsPerStep = np.int(self.parameters.xWindowSize*(1.0 - self.parameters.xOverlap) )
        numberOfYPixelsPerStep = np.int(self.parameters.yWindowSize*(1.0 - self.parameters.yOverlap) )
        

        bufferX = np.int(self.parameters.xWindowSize*(self.parameters.xOverlap))
        bufferY = np.int(self.parameters.yWindowSize*(self.parameters.yOverlap))

        numberOfXWindows = np.int((scanningXSpan - bufferX)/numberOfXPixelsPerStep)
        numberOfYWindows = np.int((scanningYSpan - bufferY)/numberOfYPixelsPerStep)
       
        windowList = []
        for yIndex in range(numberOfYWindows):
            for xIndex in range(numberOfXWindows):
                # Calculate window position
                startX = xIndex*numberOfXPixelsPerStep + self.parameters.xStart  
                endX = startX + self.parameters.xWindowSize
                startY = yIndex*numberOfYPixelsPerStep + self.parameters.yStart  
                endY = startY + self.parameters.yWindowSize
                # Append window position to list
                window_list.append(((startX, startY), (endX, endY)))
        # Return the list of windows
        return windowList
