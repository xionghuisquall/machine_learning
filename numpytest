from numpy import *
#import operator

dataSet = random.rand(4, 3)
print(dataSet)
inX = dataSet[0]
print(inX)

dataSetSize = dataSet.shape[0]

diffMat = tile(inX, (dataSetSize, 1)) - dataSet
print(diffMat)

sqDiffMat = diffMat ** 2
print(sqDiffMat)

sqDistances = sqDiffMat.sum(axis=1)
print(sqDistances)

distances = sqDistances ** 0.5
print(distances)

sortedDistIndicies = distances.argsort()
print(sortedDistIndicies)

classCount = {}
#for i in range(3) :
    # voteIlabel = labels[sortedDistIndicies[i]]