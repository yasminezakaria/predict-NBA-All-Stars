# This file contains scripts for detecting overall performance for a classifier on one season by using confusion matrix
# and under sampling by 3 times
# So for one season i see how many TP, TN, FP, FP for each file and concatenate them to calculate overall Precision,
# Recall and Accuracy. It contains doing Standard Scaling on data on pre-processing phase

from __future__ import division
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report, \
    roc_auc_score, roc_curve
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
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def predict_prob(probabilities, y_test, players_count):
    class_0_prob = []
    for i in range(len(probabilities)):
        class_0_prob.append((i, probabilities[i][1]))
    # print class_0_prob
    class_0_prob.sort(key=lambda tup: tup[1], reverse=True)
     # print results
    y_test = np.array(y_test)
    truePred = 0;
    for x in range(players_count):
        (a,b) = class_0_prob[x]
        if(y_test[a] == 1):
            truePred+=1
        # print X_test[a,:]
    return truePred

def get_top_players(X_test, coefficients, y_test, players_count, intercept):
    results = []
    # print len(coefficients)
    for a in range(len(X_test)):
        results.append((a,(np.sum(np.multiply(X_test[a, :], coefficients[0])) + intercept)))

    # print results
    results.sort(key=lambda tup: tup[1], reverse=True)
    # print results
    y_test = np.array(y_test)
    truePred = 0;
    for x in range(players_count):
        (a,b) = results[x]
        if(y_test[a] == 1):
            truePred+=1
        # print X_test[a,:]
    return truePred


def getArrayofSeasons(startSeason, endSeason):
    seasons = []
    for x in range(startSeason, endSeason + 1):
        seasons.append(x)
    return seasons


def under_sampling(filename, clf, test_season, prop):
    # read csv file
    url = "./SeasonEdit/" + filename + ".csv"

    # loading dataset into Pandas DataFrame
    data = pd.read_csv(url)
    query_equal = "Season==" + str(test_season)
    data_test = data.query(query_equal)
    data_test = data_test.reset_index(drop=True)
    data_train = data.query("Season!=" + str(test_season))
    data_train = data_train.reset_index(drop=True)

    print("length of training data", len(data_train))
    # Now make data set of normal players from train data
    normal_data = data_train[data_train["Status"] == 0]
    print("length of normal data", len(normal_data))
    star_data = data_train[data_train["Status"] == 1]
    print("length of All-star data", len(star_data))
    normal_indices_training = np.array(data_train[data_train["Status"] == 0].index)
    star_indices_training = np.array(data_train[data_train["Status"] == 1].index)
    # Choice random proportion of data train set
    Normal_indices_undersample = np.array(
        np.random.choice(normal_indices_training, (prop * len(star_indices_training)), replace=False))
    undersample_data = np.concatenate([star_indices_training, Normal_indices_undersample])
    undersample_data = data_train.iloc[undersample_data, :]

    print("the normal players proportion is :",
          len(undersample_data[undersample_data.Status == 0]) / len(undersample_data['Status']))
    print("the all-star players proportion is :",
          len(undersample_data[undersample_data.Status == 1]) / len(undersample_data['Status']))
    print("total number of record in resampled data is:", len(undersample_data['Status']))
    print("total number of all-star records in resampled data is:", len(undersample_data[undersample_data.Status == 1]))
    print("total number of normal records in resampled data is:", len(undersample_data[undersample_data.Status == 0]))
    us_data_X = undersample_data.ix[:, undersample_data.columns != "Status"]
    us_data_y = undersample_data.ix[:, undersample_data.columns == "Status"]
    test_data_X = data_test.ix[:, data_test.columns != "Status"]
    test_data_y = data_test.ix[:, data_test.columns == "Status"]
    # Scale data
    X_sca = StandardScaler().fit(us_data_X)
    us_data_X = X_sca.transform(us_data_X)
    test_data_X = X_sca.transform(test_data_X)
    clf.fit(us_data_X, us_data_y.values.ravel())
    pred = clf.predict(test_data_X)
    cnf_matrix = confusion_matrix(test_data_y, pred)
    TP = cnf_matrix[1, 1]  # no of all-star players which are predicted all-star
    TN = cnf_matrix[0, 0]  # no. of normal player which are predited normal
    FP = cnf_matrix[0, 1]  # no of normal players which are predicted all-star
    FN = cnf_matrix[1, 0]  # no of all-star players which are predicted normal
    top_players_count = np.count_nonzero(test_data_y.values.ravel())
    print top_players_count
    TP = predict_prob(clf.predict_proba(test_data_X), test_data_y, players_count=top_players_count)
    FP = top_players_count - TP
    return TP, TN, FP, FN


def graph_under_sampling():
    # clf = LogisticRegression(random_state=0, class_weight='balanced')
    clf = KNeighborsClassifier(n_neighbors=120)
    # clf = SVC(kernel='poly', random_state=0)
    # clf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight={1:1,0:2})
    # clf = ExtraTreesClassifier(n_estimators=150,class_weight={1:1,0:2})
    # clf = GradientBoostingClassifier(n_estimators=100)
    # clf = xgb.XGBClassifier(objective="binary:logistic")
    # clf = GaussianNB()
    # clf = SGDClassifier()
    # clf = LinearDiscriminantAnalysis()
    seasons = getArrayofSeasons(2000, 2017)
    for prop in range(3,4):
        classifier_accuracy = []
        classifier_recall = []
        classifier_precision = []
        classifier_f1 = []
        for x in seasons:
            print x
            guards_west = under_sampling("GuardsWesternConference", clf, x, prop)
            guards_east = under_sampling("GuardsEasternConference", clf, x, prop)
            front_west = under_sampling("FrontcourtWesternConference", clf, x, prop)
            front_east = under_sampling("FrontcourtEasternConference", clf, x, prop)
            season_TP = guards_west[0] + guards_east[0] + front_west[0] + front_east[0]
            season_TN = guards_west[1] + guards_east[1] + front_west[1] + front_east[1]
            season_FP = guards_west[2] + guards_east[2] + front_west[2] + front_east[2]
            season_FN = guards_west[3] + guards_east[3] + front_west[3] + front_east[3]
            season_accuracy = (season_TP + season_TN) / (season_TP + season_TN + season_FP + season_FN)
            season_recall = season_TP / (season_TP + season_FN)
            season_precision = season_TP / (season_TP + season_FP)
            season_f1 = 2*((season_recall*season_precision)/(season_recall+season_precision))
            classifier_f1.append(season_f1)
            classifier_recall.append(season_recall)
            classifier_precision.append(season_precision)
            classifier_accuracy.append(season_accuracy)
        # get mean and standard deviation of accuracies of each classifier
        recall_mean = np.mean(classifier_recall)
        recall_std = np.std(classifier_recall)
        precision_mean = np.mean(classifier_precision)
        precision_std = np.std(classifier_precision)
        accuracy_mean = np.mean(classifier_accuracy)
        accuracy_std = np.std(classifier_accuracy)
        f1_mean = np.mean(classifier_f1)
        f1_std = np.std(classifier_f1)
        print "----------------------------------------------------"
        print "Recall Mean = " + str(recall_mean)
        print "Precision Mean = " + str(precision_mean)
        print "Accuracy Mean = " + str(accuracy_mean)
        print "F1 Mean = " + str(f1_mean)
        print "Recall Std = " + str(recall_std)
        print "Precision Std = " + str(precision_std)
        print "Accuracy Std = " + str(accuracy_std)
        print "F1 Std = " + str(f1_std)
        # Create lists for the plot
        numbers = ['Accuracy', 'F1-score']
        x_pos = np.arange(len(numbers))
        CTEs = [accuracy_mean, f1_mean]
        error = [accuracy_std, f1_std]
        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Averaged numbers of all seasons')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(numbers)
        ax.set_title('Approach 3 with Under-sampling Logistic Regression')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        # plt.savefig('./under_sampling_'+str(prop)+'_LR.png')
        # plt.show()


graph_under_sampling()
