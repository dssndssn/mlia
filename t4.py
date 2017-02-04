#!env python3
# --*-- coding:gbk --*--

import bayes
from numpy import *

listOPosts, listClasses = bayes.loadDataSet()

myVocabList = bayes.createVocabList(listOPosts)

trainMat = []

for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)

print(p0V)
print(p1V)
print(pAb)

bayes.testingNB()

bayes.spamTest()

#bayes.spamTest()

#bayes.spamTest()