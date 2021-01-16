# This file contains scripts for detecting overall performance for a classifier on one season by using confusion matrix
# and over sampling
# So for one season i see how many TP, TN, FP, FP for each file and concatenate them to calculate overall Precision,
# Recall and Accuracy. It contains doing Standard Scaling on data on pre-processing phase


from __future__ import division

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler

file_features_columns = []


def get_most_important_features(features_weights, filename):
    # features_tuples = []
    # for i in len(features_weights):
    #     features_tuples.append((i,features_weights[i]))
    # features_tuples.sort(key=lambda tup: tup[1], reverse=True)
    featimp = pd.Series(features_weights, index=file_features_columns).sort_values(ascending=False)
    featimp.to_csv('./MIF/OS_KNN_' + filename + '.csv')
    return featimp


def update_features_weight(old_features, new_features):
    for i in range(0, len(old_features)):
        old_features[i] += new_features[i]
    return old_features


def update_f1(train_X, train_y, test_X, test_y, clf):
    accuracies = []
    columns = train_X.columns
    for i in range(len(columns)):
        column = columns[i]
        tmp_train_X = train_X.ix[:, train_X.columns == column]
        tmp_test_X = test_X.ix[:, test_X.columns == column]
        # # Scale data
        # X_sca = StandardScaler().fit(train_X)
        # os_data_X = X_sca.transform(train_X)
        # test_data_X = X_sca.transform(test_X)
        clf.fit(tmp_train_X, train_y.values.ravel())
        pred = clf.predict(tmp_test_X)
        # print column
        cnf_matrix = confusion_matrix(test_y, pred)
        TP = cnf_matrix[1, 1]  # no of all-star players which are predicted all-star
        TN = cnf_matrix[0, 0]  # no. of normal player which are predited normal
        FP = cnf_matrix[0, 1]  # no of normal players which are predicted all-star
        FN = cnf_matrix[1, 0]  # no of all-star players which are predicted normal
        # season_recall = TP / (TP + FN)
        # season_precision = TP / (TP + FP)
        # season_f1 = 2*((season_recall*season_precision)/(season_recall+season_precision))
        accuracies.append((TP+TN)/(TP+TN+FP+FN))
        # print (TP, TN, FP, FN)
    return accuracies


def predict_prob(probabilities, y_test, players_count, test_data_X):
    class_0_prob = []
    for i in range(len(probabilities)):
        class_0_prob.append((i, probabilities[i][1]))
    # print class_0_prob
    class_0_prob.sort(key=lambda tup: tup[1], reverse=True)
    print class_0_prob
    y_test = np.array(y_test)
    truePred = 0;
    for x in range(players_count):
        (a,b) = class_0_prob[x]
        if(y_test[a] == 1):
            truePred+=1
            # print test_data_X[a,:]
    return truePred



def get_top_player(X_test, coefficients, y_test, players_count):
    results = []
    for a in range(len(X_test)):
        results.append((a,(np.sum(np.multiply(X_test[a, :], coefficients)))))

    # print results
    results.sort(key=lambda tup: tup[1], reverse=True)
    print results
    y_test = np.array(y_test)
    truePred = 0
    for x in range(players_count):
        (a,b) = results[x]
        if(y_test[a] == 1):
            truePred+=1
        print X_test[a,:]
    return truePred


def getArrayofSeasons(startSeason, endSeason):
    seasons = []
    for x in range(startSeason, endSeason + 1):
        seasons.append(x)
    return seasons


def over_samplingW(filename, clf, test_season, times):
    # read csv file
    url = "./ReadyFilescsv/" + filename + ".csv"
    # url = "./2018-19 Seasons/" + filename + ".csv"
    # url = "./Edited_files/" + filename + ".csv"

    # loading dataset into Pandas DataFrame
    data = pd.read_csv(url)
    query_equal = "Season==" + str(test_season)
    # years_before = "Season >=" + str(test_season - 5) + "and Season!=" + str(test_season)
    data_test = data.query(query_equal)
    data_test = data_test.reset_index(drop=True)
    # data_test.to_csv("./data_test_2017_" +filename+".csv")
    # data_test = data_test.drop(['Season'],axis=1)
    data_train = data.query("Season!=" + str(test_season))
    # data_train = data_train.drop(['Season'],axis=1)

    # print("length of training data", len(data_train))
    # Now make data set of normal players from train data
    normal_data = data_train[data_train["Status"] == 0]
    print("length of normal data", len(normal_data))
    star_data = data_train[data_train["Status"] == 1]
    print("length of All-star data", len(star_data))
    # Now start oversampling of training data
    # means we will duplicate many times the value of all-star data
    for i in range(times):  # the number is chosen by myself on basis of number of all-star
        normal_data = normal_data.append(star_data)
    os_data = normal_data.copy()
    # print("length of oversampled data is ", len(os_data))
    # print("Number of normal players in oversampled data", len(os_data[os_data["Status"] == 0]))
    # print("No.of all-star players", len(os_data[os_data["Status"] == 1]))
    # print("Proportion of Normal data in oversampled data is ", len(os_data[os_data["Status"] == 0]) / len(os_data))
    # print("Proportion of All-star data in oversampled data is ", len(os_data[os_data["Status"] == 1]) / len(os_data))
    os_data_X = os_data.ix[:, os_data.columns != "Status"]
    os_data_y = os_data.ix[:, os_data.columns == "Status"]
    test_data_X = data_test.ix[:, data_test.columns != "Status"]
    test_data_y = data_test.ix[:, data_test.columns == "Status"]
    columns = os_data_X.columns
    global file_features_columns
    file_features_columns = columns
    # f1_scores = update_f1(os_data_X, os_data_y, test_data_X, test_data_y, clf)
    # Scale data
    X_sca = StandardScaler().fit(os_data_X)
    os_data_X = X_sca.transform(os_data_X)
    test_data_X = X_sca.transform(test_data_X)
    # clf.fit(os_data_X, os_data_y.values.ravel())
    pred = clf.predict(test_data_X)
    cnf_matrix = confusion_matrix(test_data_y, pred)
    TP = cnf_matrix[1, 1]  # no of all-star players which are predicted all-star
    TN = cnf_matrix[0, 0]  # no. of normal player which are predited normal
    FP = cnf_matrix[0, 1]  # no of normal players which are predicted all-star
    FN = cnf_matrix[1, 0]  # no of all-star players which are predicted normal
    top_players_count = np.count_nonzero(test_data_y.values.ravel())
    print top_players_count
    # TP = get_top_players(test_data_X, clf.coef_, test_data_y, top_players_count, clf.intercept_)
    TP = predict_prob(clf.predict_proba(test_data_X), test_data_y, players_count=top_players_count, test_data_X=test_data_X)
    FP = top_players_count - TP
    cm = (TP, TN, FP, FN)
    print cm
    return cm

def over_sampling(filename, clf, test_season, times):
    # read csv file
    url = "./ReadyFilescsv/" + filename + ".csv"
    # url = "./2018-19 Seasons/" + filename + ".csv"
    # url = "./Edited_files/" + filename + ".csv"

    # loading dataset into Pandas DataFrame
    data = pd.read_csv(url)
    query_equal = "Season==" + str(test_season)
    # years_before = "Season >=" + str(test_season - 5) + "and Season!=" + str(test_season)
    data_test = data.query(query_equal)
    data_test = data_test.reset_index(drop=True)
    # data_test.to_csv("./data_test_2018_" +filename+".csv")
    # data_test = data_test.drop(['Season'],axis=1)
    data_train = data.query("Season!=" + str(test_season))
    # data_train = data_train.drop(['Season'],axis=1)

    # print("length of training data", len(data_train))
    # Now make data set of normal players from train data
    normal_data = data_train[data_train["Status"] == 0]
    print("length of normal data", len(normal_data))
    star_data = data_train[data_train["Status"] == 1]
    print("length of All-star data", len(star_data))
    # Now start oversampling of training data
    # means we will duplicate many times the value of all-star data
    for i in range(times):  # the number is chosen by myself on basis of number of all-star
        normal_data = normal_data.append(star_data)
    os_data = normal_data.copy()
    # print("length of oversampled data is ", len(os_data))
    # print("Number of normal players in oversampled data", len(os_data[os_data["Status"] == 0]))
    # print("No.of all-star players", len(os_data[os_data["Status"] == 1]))
    # print("Proportion of Normal data in oversampled data is ", len(os_data[os_data["Status"] == 0]) / len(os_data))
    # print("Proportion of All-star data in oversampled data is ", len(os_data[os_data["Status"] == 1]) / len(os_data))
    os_data_X = os_data.ix[:, os_data.columns != "Status"]
    os_data_y = os_data.ix[:, os_data.columns == "Status"]
    test_data_X = data_test.ix[:, data_test.columns != "Status"]
    test_data_y = data_test.ix[:, data_test.columns == "Status"]
    columns = os_data_X.columns
    global file_features_columns
    file_features_columns = columns
    # f1_scores = update_f1(os_data_X, os_data_y, test_data_X, test_data_y, clf)
    # Scale data
    X_sca = StandardScaler().fit(os_data_X)
    os_data_X = X_sca.transform(os_data_X)
    test_data_X = X_sca.transform(test_data_X)
    clf.fit(os_data_X, os_data_y.values.ravel())
    pred = clf.predict(test_data_X)
    cnf_matrix = confusion_matrix(test_data_y, pred)
    TP = cnf_matrix[1, 1]  # no of all-star players which are predicted all-star
    TN = cnf_matrix[0, 0]  # no. of normal player which are predited normal
    FP = cnf_matrix[0, 1]  # no of normal players which are predicted all-star
    FN = cnf_matrix[1, 0]  # no of all-star players which are predicted normal
    top_players_count = np.count_nonzero(test_data_y.values.ravel())
    print top_players_count
    # TP = get_top_players(test_data_X, clf.coef_, test_data_y, top_players_count, clf.intercept_)
    TP = predict_prob(clf.predict_proba(test_data_X), test_data_y, players_count=top_players_count, test_data_X=test_data_X)
    FP = top_players_count - TP
    cm = (TP, TN, FP, FN)
    print cm
    return cm, clf


def graph_over_sampling():
    # clf = LogisticRegression(random_state=0, class_weight='balanced')
    clf = KNeighborsClassifier(n_neighbors=120)
    # clf = SVC(kernel='poly', random_state=0)
    # clf = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf = xgb.XGBClassifier(objective="binary:logistic")
    # clf = ExtraTreesClassifier(n_estimators=150, class_weight='balanced')
    # clf = GradientBoostingClassifier(n_estimators=100)
    # clf = GaussianNB()
    # clf = SGDClassifier(class_weight='balanced')
    # clf = LinearDiscriminantAnalysis()
    seasons = getArrayofSeasons(2000, 2016)
    # seasons = [2016]
    classifier_accuracy = []
    classifier_recall = []
    classifier_precision = []
    classifier_f1 = []
    # eastern_guards_features_weights = [0]*103
    # western_guards_features_weights = [0]*103
    # eastern_frontcourt_features_weights = [0]*103
    # western_frontcourt_features_weights = [0]*103
    for times in [90]:
        for x in seasons:
            print x
            # guards_east, eastG_clf = over_sampling("GuardsEasternConference", clf, x, times)
            # guards_west = over_samplingW("GuardsWesternConference", eastG_clf, x, times)
            # front_east, eastF_clf = over_sampling("FrontcourtEasternConference", clf, x, times)
            # front_west = over_samplingW("FrontcourtWesternConference", eastF_clf, x, times)
            # guards_west, westG_clf = over_sampling("GuardsWesternConference", clf, x, times)
            # guards_east,f = over_sampling("GuardsEasternConference", clf, x, times)
            # front_west, westF_clf = over_sampling("FrontcourtWesternConference", clf, x, times)
            # front_east, g = over_sampling("FrontcourtEasternConference", clf, x, times)
            guards_west, westG_clf = over_sampling("GuardsWesternConference", clf, x, times)
            guards_east = over_samplingW("GuardsEasternConference", westG_clf, x, times)
            front_west, westF_clf = over_sampling("FrontcourtWesternConference", clf, x, times)
            front_east = over_samplingW("FrontcourtEasternConference", westF_clf, x, times)
            season_TP = guards_west[0] + guards_east[0] + front_west[0] + front_east[0]
            season_TN = guards_west[1] + guards_east[1] + front_west[1] + front_east[1]
            season_FP = guards_west[2] + guards_east[2] + front_west[2] + front_east[2]
            season_FN = guards_west[3] + guards_east[3] + front_west[3] + front_east[3]
            # guards = over_sampling('guards', clf, x, times)
            # frontcourt = over_sampling('frontcourt', clf, x, times)
            # season_TP = guards[0] + frontcourt[0]
            # season_TN = guards[1] + frontcourt[1]
            # season_FP = guards[2] + frontcourt[2]
            # season_FN = guards[3] + frontcourt[3]
            season_accuracy = (season_TP + season_TN) / (season_TP + season_TN + season_FP + season_FN)
            season_recall = season_TP / (season_TP + season_FN)
            season_precision = season_TP / (season_TP + season_FP)
            season_f1 = 2*((season_recall*season_precision)/(season_recall+season_precision))
            classifier_recall.append(season_recall)
            classifier_precision.append(season_precision)
            classifier_accuracy.append(season_accuracy)
            classifier_f1.append(season_f1)
            # eastern_guards_features_weights = update_features_weight(eastern_guards_features_weights, f1_GE)
            # western_guards_features_weights = update_features_weight(western_guards_features_weights, f1_GW)
            # eastern_frontcourt_features_weights = update_features_weight(eastern_frontcourt_features_weights, f1_FE)
            # western_frontcourt_features_weights = update_features_weight(western_frontcourt_features_weights, f1_FW)
        # get mean and standard deviation of accuracies of each classifier
        recall_mean = np.mean(classifier_recall)
        recall_std = np.std(classifier_recall)
        precision_mean = np.mean(classifier_precision)
        precision_std = np.std(classifier_precision)
        accuracy_mean = np.mean(classifier_accuracy)
        accuracy_std = np.std(classifier_accuracy)
        f1_mean = np.mean(classifier_f1)
        f1_std = np.std(classifier_f1)
        # Create lists for the plot
        numbers = ['Accuracy', 'F1-sore']
        x_pos = np.arange(len(numbers))
        CTEs = [accuracy_mean, f1_mean]
        error = [accuracy_std, f1_std]
        print "Recall Mean = " + str(recall_mean)
        print "Precision Mean = " + str(precision_mean)
        print "Accuracy Mean = " + str(accuracy_mean)
        print "F1 Mean = " + str(f1_mean)
        print "Recall Std = " + str(recall_std)
        print "Precision Std = " + str(precision_std)
        print "Accuracy Std = " + str(accuracy_std)
        print "F1 Std = " + str(f1_std)

        #  # Most Important Features
        # print "------E F------"
        # print(get_most_important_features(eastern_frontcourt_features_weights, 'EF'))
        # print "------W F------"
        # print(get_most_important_features(western_frontcourt_features_weights, 'WF'))
        # print "------E G------"
        # print(get_most_important_features(eastern_guards_features_weights, 'EG'))
        # print "------W G------"
        # print(get_most_important_features(western_guards_features_weights, 'WG'))

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Averaged numbers of all seasons')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(numbers)
        ax.set_title('Test western model on eastern data')
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig('./W-E.png')
        plt.show()


graph_over_sampling()
