'''
Created on Nov 5, 2017

@author: andre
'''

import scipy
from scipy.stats import randint as sp_randint
from sklearn.svm import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import time
from DataHandling import DataHandling
from FeatureProcessing import *
import itertools

class Classifier:
    
    def __init__(self, classifierType = 'LinearSVC', randomizeSerach = False, gridSearch = False):
        self.classifierType = classifierType
        self.classifier = None
        self.trainingFeatures = None
        self.testFeatures = None
        self.trainingLabels = None
        self.testLabels = None
        self.data = DataHandling()
        self.randomizeSearch = randomizeSerach
        self.gridSearch = gridSearch
        self.SetClassifierType()
        self.featureProcressing = None 
        self.classificationIsPossible = False

    def SetRandomCVClassifier(self):
        if(self.classifierType == "LinearSVC"):
            self.classifier = LinearSVC()
            parameterDistribution = {'C': scipy.stats.expon(scale=100) }
        #    self.classifier = RandomizedSearchCV(self.classifier, param_distributions=parameterDistribution , n_iter=100)
        elif(self.classifierType == "RandomForest"):
            self.classifier = RandomForestClassifier(n_estimators=60)
            parameterDistribution = {"max_depth": [10, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(10, 100)}
        
        self.classifier = RandomizedSearchCV(self.classifier, param_distributions=parameterDistribution, n_iter=100)

    def SetClassifierType(self):
        if(self.classifierType == "LinearSVC"):
            self.classifier = LinearSVC()
            parametersRandom = {'C': scipy.stats.expon(scale=100) }
        elif(self.classifierType == "SVC RBF"):
            self.classifier = SVC(kernel="rbf", C=1000, gamma=10)
        elif(self.classifierType == "SVC Poly"):
            self.classifier = SVC(kernel="poly", gamma=0.1)
        elif(self.classifierType == "SVC Sig"):
            self.classifier = SVC(kernel="sigmoid", gamma=0.1)
        elif(self.classifierType == "RandomForest"):
            self.classifier = RandomForestClassifier(min_samples_split=10, n_estimators = 60)
        elif(self.classifierType == "DecisionTree"):
            self.classifier = DecisionTreeClassifier(min_samples_split=10)
        elif(self.classifierType == "NearestNeighbor"):
            self.classifier = KNeighborsClassifier()
        elif(self.classifierType == "NaiveBayes"):
            self.classifier = GaussianNB()
        elif(self.classifierType == "AdaBoost"):
            self.classifier = AdaBoostClassifier(learning_rate=0.01)

            
    def SetTrainingAndTestData(self, hogParameters, colorParameters, spatialParameters, colorSpace, reprocess=False, storeData =True):
        if(reprocess):
            featureProcessing = FeatureProcessing(hogParameters, colorParameters, spatialParameters, colorSpace = colorSpace)
            features, labels = featureProcessing.ComputeAllFeaturesAndLabels(storeData)
            random = 0
        else:
            random = np.random.randint(0,100)
            features, labels = self.data.LoadPreProcessedData()
            features, labels = shuffle(features, labels, random_state= random)

        self.trainingFeatures, self.testFeatures, self.trainingLabels, self.testLabels = train_test_split(features, labels, test_size=0.2, random_state=1675637+random)
        
    
    def TrainClassifier(self):
        start = time.time()
        self.classifier.fit(self.trainingFeatures, self.trainingLabels)
        end = time.time()
        print ('Time to train classifier ' + self.classifierType + " [s]: ", round(end-start,2))
    
    
    def TestClassifier(self, details =True):
        #someLabels = self.classifier.decision_function(self.testFeatures)
        predictedLabels = self.classifier.predict(self.testFeatures)
        predictedLabelsTraining = self.classifier.predict(self.trainingFeatures)
        print('Training accuracy score of classifier in % = ', round(accuracy_score(self.trainingLabels, predictedLabelsTraining), 8)*100)
        print('Test accuracy score of classifier in % = ', round(accuracy_score(self.testLabels, predictedLabels), 8)*100)
        if(details ==True):
            print('Precision, recall, F-Score () = ', precision_recall_fscore_support(self.testLabels, predictedLabels))
            print("Best parameters: ", self.classifier.best_params_)
            #print("Score: ", self.classifier.best_score_)
            #print("Estimator: ", self.classifier.best_estimator_)
            #print("Cross validation results: ", self.classifier.cv_results_)

    def SetFeatureProcessing(self, hogParameters, colorParameters, spatialParameters, colorSpace = colorSpace):
        self.featureProcessing = FeatureProcessing(hogParameters, colorParameters, spatialParameters, colorSpace = colorSpace)
        self.classificationIsPossible = True
        
    def UseClassifier(self, image):
        assert(self.classificationIsPossible == True, "Call 'SetFeatureProcessing' first")
        features = self.featureProcessing.Apply(image)
        label = self.classifier.predict(features)
        return label



def UseClassifier():
    myColorSpace = "LUV"
    myHogParameters = HogParameters(isOn = True, orientationsCount = 72, pixelsPerCell = (16,16), cellsPerBlock = (4,4), visualize = False, channel = 'ALL')
    myColorParameters = ColorParameters(isOn = True, binCount = 128)
    mySpatialParameters = SpatialParameters(isOn =True, spatialSize = (8,8))
    classifierList = "LinearSVC"
    classifier = Classifier(classifier)
    classifier.SetRandomCVClassifier()
    classifier.SetTrainingAndTestData(myHogParameters, myColorParameters, mySpatialParameters, myColorSpace, False)
    classifier.TrainClassifier()
    classifier.SetFeatureProcessing(myHogParameters, myColorParameters, mySpatialParameters, myColorSpace)
    classfiere


def OptimizeClassifiers():
    myColorSpace = "LUV"
    myHogParameters = HogParameters(isOn = True, orientationsCount = 72, pixelsPerCell = (16,16), cellsPerBlock = (4,4), visualize = False, channel = 'ALL')
    myColorParameters = ColorParameters(isOn = True, binCount = 128)
    mySpatialParameters = SpatialParameters(isOn =True, spatialSize = (8,8))
    #classifierList = ["LinearSVC",  "RandomForest",  "DecisionTree", "NearestNeighbor", "NaiveBayes", "AdaBoost", "SVC RBF", "SVC Poly", "SVC Sig"]
    #classifierList = ["LinearSVC", "SVC RBF", "RandomForest"]
    classifierList = ["LinearSVC", "RandomForest"]
    for classifier in classifierList:
        print("------------------------")
        classifier = Classifier(classifier)
        classifier.SetRandomCVClassifier()
        classifier.SetTrainingAndTestData(myHogParameters, myColorParameters, mySpatialParameters, myColorSpace, False)
        classifier.TrainClassifier()
        classifier.TestClassifier(True)      
        print("------------------------")  


def ExploreClassifiers():
    myColorSpace = "LUV"
    myHogParameters = HogParameters(isOn = True, orientationsCount = 72, pixelsPerCell = (16,16), cellsPerBlock = (4,4), visualize = False, channel = 'ALL')
    myColorParameters = ColorParameters(isOn = True, binCount = 128)
    mySpatialParameters = SpatialParameters(isOn =True, spatialSize = (8,8))
    #classifierList = ["LinearSVC",  "RandomForest",  "DecisionTree", "NearestNeighbor", "NaiveBayes", "AdaBoost", "SVC RBF", "SVC Poly", "SVC Sig"]
    #classifierList = ["LinearSVC", "SVC RBF", "RandomForest"]
    classifierList = ["LinearSVC"]
    for classifier in classifierList:
        print("------------------------")
        classifier = Classifier(classifier)
        classifier.SetTrainingAndTestData(myHogParameters, myColorParameters, mySpatialParameters, myColorSpace, False)
        classifier.TrainClassifier()
        classifier.TestClassifier(False)      
        print("------------------------")  



def ExploreFeatureProcessing():
    MyColorSpaceList = ["LUV", "HSV"]
    OrientationCountList = [9, 72]        
    PixelsPerCellList = [16]
    CellsPerBlockList = [4]
    ChannelList = ['ALL']
    
    HogParametersList = [OrientationCountList, PixelsPerCellList, CellsPerBlockList, ChannelList]
    HogParametersTupleList =  list(itertools.product(*HogParametersList))
    
    
    BinCountList = [128] 
    SpatialSizeList = [8, 16]
            
    MyColorSpace = None
    for color in MyColorSpaceList:
        MyColorSpace = color
        for hogParameters in HogParametersTupleList:
            orientation = hogParameters[0] 
            pixelsPerCellTuple = (hogParameters[1],hogParameters[1])
            cellsPerBlockTuple = (hogParameters[2],hogParameters[2])
            channelChoosen = hogParameters[3]
            for binCountElement in BinCountList:
                for spatialSizeElement in SpatialSizeList:
                    spatialSizeTuple = (spatialSizeElement, spatialSizeElement)
                    MyHogParameters = HogParameters(isOn = True, orientationsCount = orientation, pixelsPerCell = pixelsPerCellTuple, cellsPerBlock = cellsPerBlockTuple, visualize = False, channel = channelChoosen)
                    MyColorParameters = ColorParameters(isOn = True, binCount = binCountElement)
                    MySpatialParameters = SpatialParameters(isOn =True, spatialSize = spatialSizeTuple)
    
                    print("------------------------")
                    classifier = Classifier()
                    classifier.SetTrainingAndTestData(MyHogParameters, MyColorParameters, MySpatialParameters, MyColorSpace, reprocess= True, storeData = False)
                    classifier.TrainClassifier()
                    classifier.TestClassifier()      
                    print("------------------------")  


