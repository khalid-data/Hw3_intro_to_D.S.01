import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels y_pred"""
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(y_true)):

        if y_true[i] == True and y_pred[i] == True:
            TP += 1

        elif y_true[i] == True and y_pred[i] == False:
            FN += 1

        elif y_true[i] == False and y_pred[i] == True:
            FP += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return (2 * recall * precision) / (recall + precision)


def rmse(y_true, y_pred):
    """returns RMSE of regression task with true labels y_true and predicted labels y_pred"""
    sum = 0
    for i in range(y_true.size()):
        sum += (y_true[i] - y_pred[i]) ** 2

    return ((1 / y_true.size()) * sum) ** 0.5


def visualize_results(k_list, scores, metric_name, title, path):
    """plot a results graph of cross validation scores"""
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.plot(scores, k_list, width=0.4)

    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(path)
