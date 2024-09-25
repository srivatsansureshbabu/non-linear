import Config
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
import sys
import createNormalBase
import math
import numpy as np
import random
import joblib
import parse_example
import KernelFunctions
sgn = KernelFunctions.sgn



def trainMulticlass(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  correct = 0
  t = 0
  accuracies = []
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(weights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
      else:
        correct += 1
      count += 1
    accuracy = 100*testMulticlass(weights)
    accuracies.append(accuracy)
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ", accuracy)
    t += 1
  print('Max Accuracy: ' + str(max(accuracies)))
def trainMulticlassBinary(iterations,rate):
  weights = np.zeros((nTrainClasses,traindata.shape[1]))
  binaryWeights = np.copy(weights)
  correct = 0
  t = 0
  while (correct / traindata.shape[0]) != 1:
    r = list(range(traindata.shape[0]))
    random.shuffle(r)
    correct = 0
    count = 0
    for i in r:
      sample = traindata[i]
      answer = trainlabels[i]
      maxVal = -1
      guess = -1
      for m in range(nTrainClasses):
        val = kernel(binaryWeights[m],sample)
        if val > maxVal:
          maxVal = val
          guess = m 
      if guess != answer:
        weights[guess] = weights[guess] - rate*sample
        weights[answer] = weights[answer] + rate*sample
        binaryWeights = np.copy(weights)
        binaryWeights = KernelFunctions.binarizeAll(binaryWeights, 1, -1)
      else:
        correct += 1
      count += 1
    print("Iteration: ",t,"Train Accuracy: ",correct / count,"Test Accuracy: ",100*testMulticlass(binaryWeights))
    t += 1
def testMulticlass(weights):
  correct = 0
  for i in range(testdata.shape[0]):
    sample = testdata[i]
    answer = testlabels[i]
    maxVal = -1
    for m in range(nTrainClasses):
      val = kernel(weights[m],sample)
      if val > maxVal:
        maxVal = val
        guess = m
    if guess == answer:
      correct += 1
  return correct / testdata.shape[0]


directory = Config.directory 
dataset = Config.dataset
kernel = KernelFunctions.kernel
traindata, trainlabels, testdata, testlabels,nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset) 
init = int(sys.argv[1])
if init == 1:
  D = KernelFunctions.D
  traindata, trainlabels, testdata, testlabels,nTrainFeatures, nTrainClasses = KernelFunctions.load(directory,dataset) 
  
  traindata = sklearn.preprocessing.normalize(traindata,norm='l2')
  testdata = sklearn.preprocessing.normalize(testdata,norm='l2') 

  mu = Config.mu
  sigma = Config.sigma #/ 20#1 / (math.sqrt(617)) #/ 24#1 #/ (1.4)
  if Config.sparse == 1:
    createNormalBase.createSparse(D, nTrainFeatures, mu, sigma, Config.s)
  else:
    createNormalBase.create(D,nTrainFeatures,mu,sigma)
  size = int(D)
  base = np.random.uniform(0,2*math.pi,size)
  start = time.time()
  traindata = KernelFunctions.encode(traindata,base)
  assert traindata.shape[0] == trainlabels.shape[0]
  print("Encoding training time",time.time() - start)
  start = time.time()
  testdata = KernelFunctions.encode(testdata,base)
  print('Encoding testing time',time.time() - start)
  if Config.sparse == 1:
    joblib.dump(traindata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl',"wb"),compress=True)
    joblib.dump(testdata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl','wb'),compress=True)
  else:
    joblib.dump(traindata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl',"wb"),compress=True)
    joblib.dump(testdata,open('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl','wb'),compress=True)
  print(traindata.shape,trainlabels.shape)
  print(testdata.shape,testlabels.shape)
  sys.exit(0)
else:
  if Config.sparse == 1:
    traindata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'train.pkl')
    testdata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + str(int(Config.s*100)) + 'test.pkl')
  else:
    traindata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'train.pkl')
    testdata = joblib.load('../dataset/' + str(Config.directory) + '/' + str(Config.dataset) + str(Config.D) + 'test.pkl')

  if Config.binarize == 1:
    traindata = KernelFunctions.binarizeAll(traindata, 1, -1)
    testdata = KernelFunctions.binarizeAll(testdata, 1, -1)
  
  pass

if Config.binaryModel == 1:
  trainMulticlassBinary(260, Config.rate)
else:
  trainMulticlass(260,Config.rate)



