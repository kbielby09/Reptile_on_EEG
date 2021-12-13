import torch 
from torch.utils.data import Dataset, DataLoader
import random
import os
import csv
import pandas as pd

class Data(Dataset):
  def __init__(self):
    # initialize data members
    self.gestureData          = []
    self.groundTruth          = []

    self.csvTrainNames        = []
    self.csvTestNames         = []

    self.splitTrainDataNames  = []
    self.splitTrainEventNames = []
    self.splitTestDataNames   = []
    self.splitTestEventNames  = []

    self.shots                = 5

    # get list of file names
    self.csvTrainNames = os.listdir('./train')
    self.csvTestNames  = os.listdir('./test')

    self.rowCount = len(self.csvTrainNames)

    # print("Train directory: ", self.csvTrainNames)
    # print("Test directory: ", self.csvTestNames)
    # print("Row Count: ", self.rowCount)

    # TODO crate randomized data split
    self.sequentialSplit()

    # print("Train data: ", self.splitTrainDataNames)
    # print("Train events: ", self.splitTrainEventNames)
    # print("Test data: ", self.splitTestDataNames)
    # print("Test events: ", self.splitTestEventNames)
    # print("Train size: ", len(self.splitTrainEventNames))
    # print("Test size: ", len(self.splitTestEventNames))


  def sequentialSplit(self):
    # perform 90-10 train-test split 
    self.trainRowCount    = int(0.90 * self.rowCount) 
    self.testRowCount     = int(0.10 * self.rowCount)

    # for index in range(0, self.trainRowCount):
    #   if ('data' in self.csvTrainNames[index]):
    #     self.splitTrainDataNames.append(self.csvTrainNames[index])
    #   elif('events' in self.csvTrainNames[index]):
    #     self.splitTrainEventNames.append(self.csvTrainNames[index])
    #   else: 
    #     raise Exception("Train file error")

    for index in range(0, self.testRowCount):
      if ('data' in self.csvTrainNames[index]):
        self.splitTestDataNames.append(self.csvTrainNames[index])
      elif('events' in self.csvTrainNames[index]):
        self.splitTestEventNames.append(self.csvTrainNames[index])
      else: 
        raise Exception("Test file error")

    # for file in self.splitTrainEventNames:
    #   with open('./train/' + file) as trainCsvFile:
    #     trainReader = csv.reader(trainCsvFile)
    #     with open('trainClass.csv', 'w', newline='') as csvWriteFile:
    #       dataWriter = csv.writer(csvWriteFile)

    #       for row in trainReader:
    #         if row[1] != 'HandStart':
    #           if (int( row[1] ) == 1 or int( row[2] ) == 1 or int( row[3] ) == 1 or int( row[4] ) == 1 or int( row[5] ) == 1 or int( row[6] ) == 1):
    #             exclusiveClass = (((((bool(row[1]) ^ bool(row[2])) ^ bool(row[3])) ^ bool(row[4])) ^ bool(row[5])) ^ bool(row[6]))  
    #             if exclusiveClass: 
    #               dataWriter.writerow(row)
    #             else:
    #               singleClass = True
    #               dataRow = []
    #               for index in range(len(row)):
    #                 if index == 0:
    #                   dataRow.append(row[index])
    #                 else:
    #                   if int(row[index]) == 1 and singleClass:
    #                     singleClass = False
    #                     dataRow.append('1')
    #                   else:
    #                     dataRow.append('0')  
    #               dataWriter.writerow(dataRow)
    #           else:
    #             continue
    #         else:
    #           dataWriter.writerow(row)

    # get data from csv files
    # with open('trainClass.csv') as csvFile:
    #   csvReader = csv.reader(csvFile)

    #   with open('trainData.csv', 'w', newline='') as inputFile:
    #     inputWriter = csv.writer(inputFile)

    #     for row in csvReader:
    #       try:
    #         fileSubstring = row[0][:row[0].index('_', 8)]
    #       except Exception:
    #         print("substring not found")

    #       for file in self.splitTrainDataNames:
    #         if row[1] != 'HandStart':
    #           if fileSubstring in file: 

    #             with open('./train/' + file) as csvReadFile:
    #               csvDataReader = csv.reader(csvReadFile)

    #               for entry in csvDataReader:
    #                 if entry[0] == row[0]:
    #                   inputWriter.writerow(entry)

    # self.__createTestFiles__()


  def __createTestFiles__(self):

    # create label file for test data
    for file in self.splitTestEventNames:
      with open('./train/' + file) as testCSVFile:  
        testReader = csv.reader(testCSVFile)

        with open('testClass.csv', 'w', newline='') as testClassFile:
          testClassWriter = csv.writer(testClassFile)

          for row in testReader: 
            if row[1] != 'HandStart':
              if (int( row[1] ) == 1 or int( row[2] ) == 1 or int( row[3] ) == 1 or int( row[4] ) == 1 or int( row[5] ) == 1 or int( row[6] ) == 1):
                exclusiveClass = (((((bool(row[1]) ^ bool(row[2])) ^ bool(row[3])) ^ bool(row[4])) ^ bool(row[5])) ^ bool(row[6]))  
                if exclusiveClass: 
                  testClassWriter.writerow(row)
                else:
                  singleClass = True
                  dataRow = []
                  for index in range(len(row)):
                    if index == 0:
                      dataRow.append(row[index])
                    else:
                      if int(row[index]) == 1 and singleClass:
                        singleClass = False
                        dataRow.append('1')
                      else:
                        dataRow.append('0')  
                  testClassWriter.writerow(dataRow)
              else:
                continue
            else:
              testClassWriter.writerow(row)

    with open('testClass.csv') as testLabelFile:
      testLabelReader = csv.reader(testLabelFile)

      with open('testData.csv', 'w', newline='') as testDataFile:
        testDataWriter = csv.writer(testDataFile)

        for row in testLabelReader:
          try:
            fileSubstring = row[0][:row[0].index('_', 8)]
          except Exception:
            print("substring not found")

          for file in self.splitTestDataNames:
            if row[1] != 'HandStart':
              if fileSubstring in file: 

                with open('./train/' + file) as csvReadFile:
                  csvDataReader = csv.reader(csvReadFile)

                  for entry in csvDataReader:
                    if entry[0] == row[0]:
                      testDataWriter.writerow(entry)


  def randSplit(self):
    # perform 80-10 train-test-validate split 
    self.trainRowCount    = int(0.80 * self.rowCount) 
    self.testRowCount     = int(0.10 * self.rowCount)


  def getValidationIndices(self):
    return self.validateDataIndices
  
  def __len__(self):
    self.trainSize = 0 
    with open('trainData.csv') as dataset:
      dataReader = csv.reader(dataset)

      for row in dataReader:
        self.trainSize += 1

    # return self.trainSize
    return 5 # KAB TEST


  def __getitem__(self, index): # NOTE KAB for testing purposes
    trainClassFile = pd.read_csv('trainClass.csv')
    trainDataFile  = pd.read_csv('trainData.csv')

    classIndex = random.randint(0, 6)
    dataIndex = 0
    if classIndex == 0:
      dataIndex = random.randint(0, 151)
    elif classIndex == 1:
      dataIndex = random.randint(151, 301)
    elif classIndex == 2:
      dataIndex = random.randint(301, 401)
    elif classIndex == 3:
      dataIndex = random.randint(401, 460)
    elif classIndex == 4:
      dataIndex = random.randint(460, 610)
    elif classIndex == 5:
      dataIndex = random.randint(610, 683)

    data = list(trainDataFile.loc[dataIndex][1:])
    labelIndex = list(trainClassFile.loc[dataIndex][1:])
    labelIndex = labelIndex.index(1)
    label = []
    label.append(labelIndex) 

    dataTensor  = torch.Tensor(data)
    truthTensor = torch.Tensor(label)

    # print("truthTensor: ", truthTensor)
    # print("data: ", dataTensor)
  
    return dataTensor, truthTensor

  # function returns batch of shots for all classes 
  # def __getitem__(self, index):  
  #   data = []
  #   label = []

    # class1Data = []
    # class2Data = []
    # class3Data = []
    # class4Data = []
    # class5Data = []
    # class6Data = []

    # trainClassFile = pd.read_csv('trainClass.csv')
    # trainDataFile  = pd.read_csv('trainData.csv')

    # while len(class1Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 1:
    #     class1Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 1 data: ", class1Data)

    # while len(class2Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 2:
    #     class2Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 2 data: ", class2Data)

    # while len(class3Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 3:
    #     class3Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 3 data: ", class3Data)

    # while len(class4Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 4:
    #     class4Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 4 data: ", class4Data)

    # while len(class5Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 5:
    #     class5Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 5 data: ", class5Data)


    # while len(class6Data) != 5:
    #   randomIndex = random.randint(0, len(trainClassFile) - 1)
      
    #   # get list of hand gestures that occurred
    #   classification = list(trainClassFile.loc[randomIndex])

    #   if classification.index(1) == 6:
    #     class6Data.append(list(trainDataFile.loc[randomIndex])[1:])

    # # print("Class 6 data: ", class6Data)

    # data = []
    # data.append(class1Data)
    # data.append(class2Data)
    # data.append(class3Data)
    # data.append(class4Data)
    # data.append(class5Data)
    # data.append(class6Data)

    # label = [1,2,3,4,5,6]


    # with open("trainData.csv") as trainData:
    #   trainDataReader = csv.reader(trainData)

    #   with open('trainClass.csv') as trainClass:
    #     trainClassReader = csv.reader(trainClass) 

    #     rowNumber = 0
    #     for row in trainDataReader:
    #       rowNumber += 1
    #       if index == rowNumber:
    #         data = row[1:]

    #         for dataindex in range(len(data)):
    #           data[dataindex] = int(data[dataindex])
            
    #         break

    #     rowNumber = 0
    #     for row in trainClassReader:
    #       rowNumber += 1
    #       if index == rowNumber:
    #         label = row[1:]

    #         for labelindex in range(len(label)):
    #           label[labelindex] = int(label[labelindex])

    #         index = label.index(1)
    #         label = []
    #         label.append(index)
            
    #         break
    
    # get data row from pandas dataframe
    # data = list(trainDataFile.loc[index][1:])
    # labelIndex = list(trainClassFile.loc[index][1:])
    # labelIndex = labelIndex.index(1)
    # label = []
    # label.append(labelIndex) 
    # label = list(trainClassFile.loc[index][1:])

    # transform data and label to tensors
    # dataTensor  = torch.Tensor(data)
    # truthTensor = torch.Tensor(label)

    # print("truthTensor: ", truthTensor)
    # # print("data: ", dataTensor)
  
    # return dataTensor, truthTensor

class ReptileNet(torch.nn.Module):
  def __init__(self):
    super(ReptileNet, self).__init__()
    # self.conv1 = torch.nn.Conv2d(6, 32, 5, 1)
    self.fc1   = torch.nn.Linear(32, 64)
    # self.fc1   = torch.nn.Linear(896, 64)
    self.fc2   = torch.nn.Linear(64, 32)
    # self.fc3   = torch.nn.Linear(32, 1)
    # self.fc3   = torch.nn.Linear(32, 6)
    self.fc3   = torch.nn.Linear(32, 32)
    self.fc4   = torch.nn.Linear(32, 128)
    self.fc5   = torch.nn.Linear(128, 6)
    
  def forward(self, x):
    # print("shape: ", x.size())
    # x = torch.nn.functional.relu(self.conv1(x))
    # x = torch.nn.functional.relu(self.fc0(x))
    # x = torch.flatten(x, 1)
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = torch.nn.functional.relu(self.fc3(x))
    x = torch.nn.functional.relu(self.fc4(x))
    x = torch.nn.functional.relu(self.fc5(x))

    x = torch.softmax(x, dim=-1)
    return x


class Metrics:
  def __init__(self):
    self.falsePositives = 0
    self.falseNegatives = 0
    self.truePositives  = 0
    self.trueNegatives  = 0
    self.Recall         = 0
    self.F1             = 0
    self.Precision      = 0
    self.Accuracy       = 0

  def calculatedRecall(self):
    denom = self.truePositives + self.falseNegatives
    if denom != 0:
      self.Recall = self.truePositives / (denom)
    else:
      Recall = 0

  def addResult(self, result):
    if(result == 'true positive'):
      self.truePositives += 1
    elif(result == 'true negative'):
      self.trueNegatives += 1
    elif(result == 'false positive'):
      self.falsePositives += 1
    elif(result == 'false negative'):
      self.falseNegatives += 1
    else:
      print("invalid entry")
    
    #calculate all metrics
    self.calculateF1()
    self.calculateAccuracy()
    self.calculatePrecision()
  
  def calculateF1(self):
    self.calculatedRecall()
    self.calculatePrecision()
    denom = self.Precision + self.Recall
    if denom != 0:
      self.F1 = 2 * (self.Precision * self.Recall) / (self.Precision + self.Recall)
    else:
      self.F1 = 0

  def calculateAccuracy(self):
    self.Accuracy = (self.truePositives + self.trueNegatives) / (self.truePositives + self.trueNegatives + self.falseNegatives + self.falsePositives)

  def calculatePrecision(self):
    denom = self.truePositives + self.falsePositives
    if denom != 0:
      self.Precision = self.truePositives / (denom)
    else:
      self.Precision = 0

  def getTruePositives(self):
    return self.truePositives

  def getTrueNegatives(self):
    return self.trueNegatives

  def getFalsePostives(self):
    return self.falsePositives

  def getFalseNegatives(self):
    return self.falseNegatives

  def getRecall(self):
    return self.Recall

  def getF1(self):
    return self.F1

  def getPrecision(self):
    return self.Precision

  def getAccuracy(self):
    return self.Accuracy

class Reptile:
  def __init__(self):
    self.model = ReptileNet()
    self.__setHyperparameters__()
    # self.criterion = torch.nn.CrossEntropyLoss()
    self.criterion = torch.nn.MultiLabelSoftMarginLoss()
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
    
    # create metrics instance for hyperparameter tuning
    self.metrics = Metrics()

  def train(self, dataLoader):
    print("training")
    for iteration in range(self.niterations):

      oldStateDict = self.model.state_dict()

      for epoch in range(self.innerepochs):
        for i, data in enumerate(dataLoader, 0):
          inputs, labels = data

          self.model.zero_grad()
          self.output = self.model(inputs)
          self.output.squeeze(0)
          # print("output shape: ", self.output.size())
          loss = self.criterion(self.output, labels)
          loss.backward()

          # update model parameters according to Reptile algorithm
          for param in self.model.parameters(True):
            param.data -= self.innerstepsize * param.grad.data 

      newStateDict = self.model.state_dict()

      outerStepSize = self.outerstepsize0 * (1 - iteration / self.niterations)

      self.model.load_state_dict({name : oldStateDict[name] + (newStateDict[name] - oldStateDict[name]) * outerStepSize for name in oldStateDict})

      

  def predict(self, data):
    old_model_weights = self.model.state_dict()
    predict = self.model(data)
    self.model.load_state_dict(old_model_weights)
    return predict
  
  def __setHyperparameters__(self):
    self.innerstepsize = 0.02 # stepsize in inner SGD
    self.innerepochs = 4 # number of epochs of each inner SGD
    self.outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
    self.niterations = 10 # number of outer updates; each iteration we sample one task and update on it

    self.learning_rate = 0.003

    self.train_shots = 20
    self.shots = 5
    self.classes = 5

  def TuneHyperParameters(self, dataloader):
    print("Tuning hyperparameters")

    # save old F1
    oldF1 = self.metrics.getF1()

    # reset model weights
    for child in self.model.children():
      child.reset_parameters()

    

    # retrainModel
    self.train(dataloader)

    # get the test data as pandas dataframe
    testData = pd.read_csv('testData.csv')
    testClass = pd.read_csv('testClass.csv')

    # generate test data values
    for index in range(1, 151):
      classIndex = random.randint(0, 6)
      dataIndex = 0
      if classIndex == 0:
        dataIndex = random.randint(0, 151)
      elif classIndex == 1:
        dataIndex = random.randint(151, 301)
      elif classIndex == 2:
        dataIndex = random.randint(301, 374)
      elif classIndex == 3:
        dataIndex = random.randint(374, 440)
      elif classIndex == 4:
        dataIndex = random.randint(440, 590)
      elif classIndex == 5:
        dataIndex = random.randint(590, 740)

      testDataList = []
      testDataList.append(list(testData.loc[dataIndex])[1:])

      testDataList = torch.Tensor(testDataList)

      data = self.predict(testDataList)

      # get actual label for data
      testClassList = []
      testClassList = (list(testClass.loc[dataIndex])[1:])

      actualIndex = testClassList.index(1)

      # print("output: ", data)

      value, predicted = torch.max(data, 1)

      predicted = predicted.numpy()
      value = value.detach().numpy()

      print("predicted: ", predicted[0], " actual: ", actualIndex, " value: ", value)

      if predicted[0] == actualIndex and value[0] > 0.70:
        self.metrics.addResult('true positive')
      elif predicted[0] == actualIndex and value[0] < 0.70:
        self.metrics.addResult('false negative')
      elif predicted[0] != actualIndex and value[0] > 0.70:
        self.metrics.addResult('false positive')
      elif predicted[0] != actualIndex and value[0] < 0.70:
        self.metrics.addResult('true negative')

    print('False positive: ', self.metrics.getFalsePostives())
    print('True positives: ', self.metrics.getTruePositives())
    print('True Negative: ', self.metrics.getTrueNegatives())
    print('False Negatives: ', self.metrics.getFalseNegatives())
    print('Precision: ', self.metrics.getPrecision())
    print('Accuracy: ', self.metrics.getAccuracy())
    print('F1: ', self.metrics.getF1())
    print('Recall: ', self.metrics.getRecall())

    newF1 = self.metrics.getF1()

    if (newF1 - oldF1) > 0.001 or newF1 == 0 or (newF1 - oldF1) < 0:
      print("learning rate: ", self.learning_rate)
      if (newF1 - oldF1) > 0.001 : 
        # increase learning rate
        self.learning_rate += 0.001
        torch.save(self.model.state_dict(), './modelWeights.pt')
      self.TuneHyperParameters(dataloader)      
    else: 
      print("end of tuning")
      

if __name__ == "__main__":
  trainData = Data() 
  # trainData.__getitem__(5)

  dataloader = DataLoader(trainData, batch_size=1, shuffle=True, num_workers=1)
  reptile = Reptile()
  reptile.train(dataloader)

  reptile.TuneHyperParameters(dataloader)

  