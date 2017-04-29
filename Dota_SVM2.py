from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random

from random import randint

xx = [[0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
yy = [0, 0, 0, 1, 0, 1]



# make a 100x1 array to hold the corrsponding wins/loss for the 100 hero selections
def makeDummyTestTarget():
    target = []
    for i in range(100):
        target.append(randint(0, 1))
    return target

# make a 226x1 array to hold the 10 hero selections (5 per team)
def makeDummyInputHerosArray():
    f = []
    for i in range(226):
        f.append(0)

    rand1 = random.sample(range(1, 113), 5)
    rand2 = random.sample(range(114, 226), 5)

    for i in range(0,5):
        f[rand1[i]] = 1
        f[rand2[i]] = 1

    return f


if __name__ == '__main__':
    print("hii")

    # make 100 hero selection samples
    X = []
    for i in range(0,100):
        a = makeDummyInputHerosArray()
        X.append(a)

    # 100 targer win/loss for the 100 hero selection samples
    y = makeDummyTestTarget()

    clf = svm.SVC(gamma=0.001, C=100, )

    # test code WORKS
    clf.fit(xx, yy)
    print(clf.predict([0, 1, 1, 0]))

    # pedict with a sample hero selection
    clf.fit(X, y)
    test = makeDummyInputHerosArray()
    print(clf.predict(test))
