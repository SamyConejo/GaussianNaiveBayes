from statistics import stdev, mean
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from NaiveBayes import GaussianNaiveBayes
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

def init_bayes(X, y):
    nb_acc = []
    nb_recall = []
    nb_precision = []
    nb_auc = []

    nb0_recall = []
    nb0_precision = []


    # set stratified 10-folds CV
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    for train_index, test_index in skf.split(X, y):

        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        nb = GaussianNaiveBayes()
        nb.fit(X_train_fold.values, y_train_fold.values.ravel())
        y_prediction = nb.predict(X_test_fold.values)
        pred_prob = nb.predict_proba(X_test_fold.values)
        print('pred ', pred_prob)

        nb_acc.append(accuracy_score(y_test_fold, y_prediction))
        nb_recall.append(recall_score(y_test_fold, y_prediction))
        nb_precision.append(precision_score(y_test_fold, y_prediction))

        nb_auc.append(roc_auc_score(y_test_fold, pred_prob[:, 1]))

    print('---------- BAYES -------------')
    print('Overall BAYES  Accuracy:', mean(nb_acc))
    print('Standard Deviation is:', stdev(nb_acc))
    print('-----------------------')
    print('Overall BAYES Recall:', mean(nb_recall))
    print('Standard Deviation is:', stdev(nb_recall))
    print('-----------------------')
    print('Overall BAYES Precision:', mean(nb_precision))
    print('Standard Deviation is:', stdev(nb_precision))
    print('-----------------------')
    print('Overall BAYES AUC:', mean(nb_auc))
    print('Standard Deviation is:', stdev(nb_auc))


if __name__ == '__main__':

    datos = pd.read_csv('dataset.csv')

    # dataset original
    df = pd.DataFrame(datos, columns=['class', '42', '40', '6', '24', '14', '4', '10', '22'])
    df.to_csv('DATA.csv')

    # dataset normalizado
    scaler = preprocessing.MinMaxScaler()
    values = df.values
    scaledValues = scaler.fit_transform(values)
    df = pd.DataFrame(scaledValues, columns=[0, 42, 40, 6, 24, 14, 4, 10, 22])
    df.to_csv('DATA_normalized.csv')

    # our hipothesys
    X = df[[42, 40, 6, 24, 14, 4, 10, 22]]
    y = df[[0]]

    # init bayes classificator
    init_bayes(X,y)



