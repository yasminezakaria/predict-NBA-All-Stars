from __future__ import division

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import random as random
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# filename = "GuardsEasternConference"
filename = "FrontcourtWesternConference"
# read csv file
url = "./ReadyFilescsv/" + filename + ".csv"
# url = "./PCAData/" + filename + ".csv"

# loading dataset into Pandas DataFrame
df = pd.read_csv(url, skip_blank_lines=True)
# Save Season Data that will be tested
testSeason = df.query('Season==2010')
print testSeason.query('Status == 1')
allStarTestCount = len(testSeason.query('Status == 1'))
y_test = testSeason.iloc[:, 102]
testSeason = testSeason.drop('Status', axis=1)
X_test = testSeason.iloc[:, :]

# Training seasons data
train = df.query('Season!=2010')
train = train.query('Season >= 2000')
y_train = train.iloc[:, 102]
train = train.drop('Status', axis=1)
X_train = train.iloc[:, :]
columns = X_train.columns

# Scale data
# X_sca = StandardScaler()
# X = X_sca.fit_transform(X)

X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)
X_train = np.array(X_train)
# print X_test

X_trainAllStar = []
X_trainRegular = []
for x in range(len(X_train)):
    if y_train[x] == 0:
        X_trainRegular.append(list(X_train[x]))
    else:
        X_trainAllStar.append(list(X_train[x]))
        # X_trainAllStar.append(list(X_trainAll[x]))

# number of all-star players in file
all_starPlayersCount = len(X_trainAllStar)
regularPlayersCount = len(X_trainRegular)
numOfIterations = regularPlayersCount / all_starPlayersCount
totalTrainingRecords = all_starPlayersCount + regularPlayersCount
print all_starPlayersCount
"""Different Classifier USed LR - LDA - SGD"""
# clf = RandomForestClassifier(n_estimators=104)
# clf = LogisticRegression(random_state=0, class_weight='balanced')
# clf = LinearDiscriminantAnalysis()
clf = SGDClassifier(class_weight='balanced')
# clf = SVC(kernel='poly', random_state=0)
coefficients = []
intercept = []
for x in range(500):
    # Choose regular players randomly
    X_set = random.sample(X_trainRegular, (all_starPlayersCount)) + X_trainAllStar
    y_set = [0]*(all_starPlayersCount) + [1]*(all_starPlayersCount)

    # Start to learn
    clf.fit(X_set,y_set)
    coefficients.append(clf.coef_)
    intercept.append(clf.intercept_)
    # print clf.get_params()

# Average collected coefs and n0
coefficients = np.array(coefficients)
averagedCoef = np.mean(coefficients, axis=0)
# print intercept
averagedIntercept = np.mean(intercept, axis=0)
# print averagedIntercept

clf.coef_ = averagedCoef
clf.intercept_ = averagedIntercept
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# print y_pred
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix= "+str(cm))
print ("Recall/Sensitivity = " + str(recall_score(y_test,y_pred)))
print ("Precision = " + str(precision_score(y_test,y_pred)))
print (classification_report(y_test, y_pred))

# getting top 3/2 players only
results = []
for a in range(len(X_test)):
    results.append((a,(np.sum(np.multiply(X_test[a, :], coefficients)))))

# print results
results.sort(key=lambda tup: tup[1], reverse=True)
# print results

truePred = 0
for x in range(3):
    (a,b) = results[x]
    if(y_test[a] == 1):
        truePred+=1
    # print X_test[a,:]
print truePred/allStarTestCount

