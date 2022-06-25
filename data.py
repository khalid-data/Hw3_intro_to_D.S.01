import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)

def load_data(path):
    """
    reads and returns the pandas DataFrame
    :param path: path to csv file
    :return: the data frame
    """
    df = pd.read_csv(path)
    return df

def adjust_labels(y):
    """ make new one"""
    """adjust labels of season from {0,1,2,3} to {0,1}"""
    n_y = []
    for i in range(len(y)):
        if y[i] == 1 or y[i] == 0:
            n_y.append(0)
        elif y[i] == 2 or y[i] == 3:
            n_y.append(1)
    return n_y


def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


def mean(values):
    """
    calculate the mean of values
    :param values:list that contains numbers
    :return: the mean
    """
    temp = sum1(values)
    return temp / len(values)

def std(list_of_values):
    """
    this function calculates the variance of a list of data
    :param list_of_values: the list of data we want to calculate its variance
    :return: the variance
    """
    average = mean(list_of_values)
    squared_sum = sum1([(x - average) ** 2 for x in list_of_values])
    return (squared_sum / (len(list_of_values) - 1)) ** (1 / 2)

def sum1(values):
    """
    calculates the sum of values(count is sum)
    :param values: list that contains numbers
    :return: the sum
    """
    count = 0
    for i in values:
        count += i
    return count


class StandardScaler():
    def __init__(self):
        """object instantiation"""
        """self.data = np.array size 2, num_of_features first row saves mean second saves stdv"""
        self.statistics = np.zeros(shape=(2, 4))

    def fit(self, X):
        """fit scaler by learning mean and standard deviation per feature """
        for col in range(X.shape[1]):
            self.statistics[0][col] = mean(X[:, col])
            self.statistics[1][col] = std(X[:, col])
        return self

    def transform(self, X):
        """transform X by learned mean and standard deviation, and return it """
        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                X[row][col] = (X[row][col] - self.statistics[0][col]) / self.statistics[1][col]
        return X

    def fit_transform(self, X):
        """fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)
