# This file contains scripts for detecting overall performance for a classifier on one season by using confusion matrix
# and over sampling using SMOTEENN libaray
# So for one season i see how many TP, TN, FP, FP for each file and concatenate them to calculate overall Precision,
# Recall and Accuracy. It contains doing Standard Scaling on data on pre-processing phase


from __future__ import division

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb


file_features_columns = []


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()


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

def get_most_important_features(features_weights, filename):
    # features_tuples = []
    # for i in len(features_weights):
    #     features_tuples.append((i,features_weights[i]))
    # features_tuples.sort(key=lambda tup: tup[1], reverse=True)
    featimp = pd.Series(features_weights, index=file_features_columns).sort_values(ascending=False)
    featimp.to_csv('./MIF/KNN' + filename + '.csv')
    return featimp


def update_features_weight(old_features, new_features):
    for i in range(0, len(old_features)):
        old_features[i] += new_features[i]
    return old_features


def get_top_players(X_test, coefficients, y_test, players_count, intercept):
    results = []
    # print len(coefficients)
    for a in range(len(X_test)):
        results.append((a,(np.sum(np.multiply(X_test[a, :], coefficients[0])) + intercept)))

    # print results
    results.sort(key=lambda tup: tup[1], reverse=True)
    # print results
    y_test = np.array(y_test)
    truePred = 0
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


def over_sampling_SMOTE(filename, clf, test_season):
    # read csv file
    # url = "./combined_files/" + filename + ".csv"
    # url = "./ReadyFilescsv/" + filename + ".csv"
    url = "./SeasonEdit/" + filename + ".csv"

    # loading dataset into Pandas DataFrame
    data = pd.read_csv(url)
    query_equal = "Season==" + str(test_season)
    years_before = "Season >=" + str(test_season - 10) + "and Season <" + str(test_season)
    not_test_season = "Season!=" + str(test_season)
    data_test = data.query(query_equal)
    # if(filename == "FrontcourtWesternConference"):
    #     data_test = data_test.drop(['Tm_Rank'], axis=1)
    # if(filename == "FrontcourtEasternConference"):
    #     data_test = data_test.drop(['Tm_Rank', 'Season'], axis=1)
    # data_test = data_test.drop(['Season'],axis=1)
    # # print data_test.head(2)
    data_train = data.query(not_test_season)
    # if(filename == "FrontcourtWesternConference"):
    #     data_train = data_train.drop(['Tm_Rank'], axis=1)
    # if(filename == "FrontcourtEasternConference"):
    #     data_train = data_train.drop(['Tm_Rank', 'Season'], axis=1)
    # data_train = data_train.drop(['Season'],axis=1)

    print("length of training data", len(data_train))
    # Now make data set of normal players from train data
    normal_data = data_train[data_train["Status"] == 0]
    print("length of normal data", len(normal_data))
    star_data = data_train[data_train["Status"] == 1]
    print("length of All-star data", len(star_data))
    # os = SMOTE(random_state=10,ratio=0.6,sampling_strategy='minority')  # We are using SMOTE as the function for oversampling
    os = SMOTEENN(random_state=0, sampling_strategy='minority')
    # os = SMOTETomek()
    data_train_X = data_train.ix[:, data_train.columns != "Status"]
    data_train_y = data_train.ix[:, data_train.columns == "Status"]
    columns = data_train_X.columns
    global file_features_columns
    file_features_columns = columns
    os_data_X, os_data_y = os.fit_sample(data_train_X, data_train_y.values.ravel())
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=["Status"])
    # we can Check the numbers of our data
    print("length of oversampled data is ", len(os_data_X))
    print("Number of normal players in oversampled data", len(os_data_y[os_data_y["Status"] == 0]))
    print("No.of All-star ", len(os_data_y[os_data_y["Status"] == 1]))
    print("Proportion of Normal players in oversampled data is ", len(os_data_y[os_data_y["Status"] == 0]) / len(os_data_X))
    print("Proportion of all-star players in oversampled data is ", len(os_data_y[os_data_y["Status"] == 1]) / len(os_data_X))
    test_data_X = data_test.ix[:, data_test.columns != "Status"]
    test_data_y = data_test.ix[:, data_test.columns == "Status"]
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
    # top_players_count = 2 if (filename.startswith("Guards")) else 3
    top_players_count = np.count_nonzero(test_data_y.values.ravel())
    print top_players_count
    # TP = get_top_players(test_data_X, clf.coef_, test_data_y, top_players_count, clf.intercept_)
    TP = predict_prob(clf.predict_proba(test_data_X), test_data_y, players_count=top_players_count)
    FP = top_players_count - TP
    cm = (TP, TN, FP, FN)
    return cm#, np.array(clf.coef_[0])


def graph_over_sampling():
    # clf = LogisticRegression(random_state=0, class_weight='balanced')
    clf = KNeighborsClassifier(n_neighbors=120)
    # clf = SVC(kernel='Poly', random_state=0)
    # clf = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf = GaussianNB()
    # clf = SGDClassifier()
    # clf = xgb.XGBClassifier(objective="binary:logistic")
    # clf = ExtraTreesClassifier(n_estimators=150, class_weight='balanced')
    # clf = GradientBoostingClassifier(n_estimators=100)
    # clf = LinearDiscriminantAnalysis()
    seasons = getArrayofSeasons(2000, 2017)
    classifier_accuracy = []
    classifier_recall = []
    classifier_precision = []
    classifier_f1 = []
    # eastern_guards_features_weights = [0]*102
    # western_guards_features_weights = [0]*102
    # eastern_frontcourt_features_weights = [0]*102
    # western_frontcourt_features_weights = [0]*102
    for x in seasons:
        print x
        guards_west = over_sampling_SMOTE("GuardsWesternConference", clf, x)
        guards_east = over_sampling_SMOTE("GuardsEasternConference", clf, x)
        front_west = over_sampling_SMOTE("FrontcourtWesternConference", clf, x)
        front_east = over_sampling_SMOTE("FrontcourtEasternConference", clf, x)
        # guards_west, coef_guards_west = over_sampling_SMOTE("GuardsWesternConference", clf, x)
        # guards_east, coef_guards_east = over_sampling_SMOTE("GuardsEasternConference", clf, x)
        # front_west, coef_front_west = over_sampling_SMOTE("FrontcourtWesternConference", clf, x)
        # front_east, coef_front_east = over_sampling_SMOTE("FrontcourtEasternConference", clf, x)
        season_TP = guards_west[0] + guards_east[0] + front_west[0] + front_east[0]
        season_TN = guards_west[1] + guards_east[1] + front_west[1] + front_east[1]
        season_FP = guards_west[2] + guards_east[2] + front_west[2] + front_east[2]
        season_FN = guards_west[3] + guards_east[3] + front_west[3] + front_east[3]
        # season_TP = guards_east[0]
        # season_TN = guards_east[1]
        # season_FP = guards_east[2]
        # season_FN = guards_east[3]
        # guards = over_sampling_SMOTE('guards', clf, x)
        # frontcourt = over_sampling_SMOTE('frontcourt', clf, x)
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
        print str(season_recall) +", "+ str(season_f1)
        # eastern_guards_features_weights = update_features_weight(eastern_guards_features_weights, coef_guards_east)
        # western_guards_features_weights = update_features_weight(western_guards_features_weights, coef_guards_west)
        # eastern_frontcourt_features_weights = update_features_weight(eastern_frontcourt_features_weights, coef_front_east)
        # western_frontcourt_features_weights = update_features_weight(western_frontcourt_features_weights, coef_front_west)
    # get mean and standard deviation of accuracies of each classifier
    recall_mean = np.mean(classifier_recall)
    recall_std = np.std(classifier_recall)
    precision_mean = np.mean(classifier_precision)
    precision_std = np.std(classifier_precision)
    accuracy_mean = np.mean(classifier_accuracy)
    accuracy_std = np.std(classifier_accuracy)
    f1_mean = np.mean(classifier_f1)
    f1_std = np.std(classifier_f1)
    print "Recall Mean = " + str(recall_mean)
    print "Precision Mean = " + str(precision_mean)
    print "Accuracy Mean = " + str(accuracy_mean)
    print "Recall Std = " + str(recall_std)
    print "Precision Std = " + str(precision_std)
    print "Accuracy Std = " + str(accuracy_std)
    print "F1 Mean = " + str(f1_mean)
    print "F1 Std = " + str(f1_std)

    # # Most Important Features
    # print "------E F------"
    # print(get_most_important_features(eastern_frontcourt_features_weights, 'EF'))
    # print "------W F------"
    # print(get_most_important_features(western_frontcourt_features_weights, 'WF'))
    # print "------E G------"
    # print(get_most_important_features(eastern_guards_features_weights, 'EG'))
    # print "------W G------"
    # print(get_most_important_features(western_guards_features_weights, 'WG'))

    # # Create lists for the plot
    numbers = ['Accuracy','F1']
    x_pos = np.arange(len(numbers))
    CTEs = [accuracy_mean, f1_mean]
    error = [accuracy_std, f1_std]
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Averaged numbers of all seasons')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(numbers)
    ax.set_title('Approach 4 with SMOTEENN oversampling KNN')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('./SMOTEmean_KNN.png')
    # # plt.show()


graph_over_sampling()
# over_sampling_SMOTE("FrontcourtEasternConference", LogisticRegression(class_weight='balanced', random_state=0), 2015)
