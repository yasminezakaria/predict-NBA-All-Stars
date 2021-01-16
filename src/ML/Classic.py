from __future__ import division

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns



# filename = "GuardsEasternConference"
filename = "FrontcourtEasternConference"
# read csv file
url = "./ReadyFilescsv/"+filename+".csv"
# url = "./PCAData/" + filename + ".csv"


"""loading dataset into Pandas DataFrame"""
df = pd.read_csv(url)
# print df.info()
sns.countplot("Status",data=df)
y = df.iloc[:, 102]
df = df.drop('Status', axis=1)
X = df.iloc[:, :]

""""Scale data""""
X_sca = StandardScaler()
X = X_sca.fit_transform(X)

"""PCA to decrease Features Dimensionality"""
# pca = PCA(n_components=4)
# X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#,random_state=104)


# print len(df)

X_train = np.array(X_train)
X_test = np.array(X_test)


"""Different Machine Learning Classifiers tested on the data set"""
"""-----------------------------------------------------------------------"""


"""Running Logistic Regression"""
clf = LogisticRegression(random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# print str(clf.coef_)


"""Running Naive Bayes"""
# clf = GaussianNB()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

"""Running Stochastic Gradient Descent"""
# clf = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101,early_stopping=True,validation_fraction=0.2,n_iter_no_change=50)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

"""Running KNN"""
# clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # p = 1 is manhattan distance, p = 2 is Euclidean
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

"""Running SVM"""

# clf = SVC(kernel='linear', random_state=0).fit(X_train, y_train)
# print clf
# clf = SVC(kernel='rbf', random_state=0).fit(X_train, y_train)
# clf = SVC(kernel='poly', random_state=0).fit(X_train, y_train)
# y_pred = clf.predict(X_test)

"""Running Decision Tree"""
# clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print clf

"""Random Forest"""
# clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf.fit(X, y)
# y_pred = clf.predict(X_test)




# output = pd.DataFrame(columns=['X_test1', 'X_test2', 'y_pred','y_test'])
# output['y_pred'] = y_pred
# output['X_test1'] = X_test['principal component 1']
# output['X_test2'] = X_test['principal component 2']
# output['y_test'] = y_test
#
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix= "+str(cm))
print ("Recall/Sensitivity = " + str(recall_score(y_test,y_pred)))
print ("Precision = " + str(precision_score(y_test,y_pred)))

print clf.score(X_test, y_test)


"""Some illustrating graphs for test and train sets"""

# X_set, y_set = X_test, y_pred
# # generates every pixel in the table. MeshGrid creates one entry for every point from X1 to X2
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))
# # classifies every pixel as 0 or 1
# plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j, edgecolor='black')
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# plt.legend()
# plt.title('Logistic Regression Testing Data')
# plt.xlabel('principal component 1')
# plt.ylabel('principal component 2')
# plt.savefig("./figTest.png")
# plt.show()




# X_set, y_set = X_train, y_train
# # generates every pixel in the table. MeshGrid creates one entry for every point from X1 to X2
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))
# # classifies every pixel as 0 or 1
# plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j, edgecolor='black')
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# plt.legend()
# plt.title('Logistic Regression Training Data')
# plt.xlabel('principal component 1')
# plt.ylabel('principal component 2')
# plt.savefig("./figTrain.png")


