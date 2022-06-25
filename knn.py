import numpy as np
import scipy.stats
from abc import abstractmethod
from data import StandardScaler


class KNN:
    def __init__(self, k):
        """object instantiation, save k and define a scaler object"""
        self._points = []
        self._k = k
        self._labels = []
        self._scaler = StandardScaler()

    def fit(self, X_train, y_train):
        """fit scaler and save X_train and y_train"""
        self._points = StandardScaler.fit_transform(self._scaler, X_train)
        self._labels = y_train

    def get_scaler(self):
        return self._scaler

    @abstractmethod
    def predict(self, X_test):
        pass

    def neighbours_indices(self, x):
        """for a given point x, find indices of k closest points in the training set"""

        """# dict where in each index key we get the dist between x and the point from train (that has the same index)
        index_to_distance = {point_index: self.dist(p, x) for point_index, p in enumerate(self._points)}

        # sort and get closest k
        sorted_indices = sorted(index_to_distance.keys(), key=lambda t: index_to_distance[t])

        # count the frequency of each label in the closest k points
        closest_indices = [sorted_indices[i] for i in sorted_indices[:self._k]]

       

        return closest_indices"""
        distances = [self.dist(x, point) for point in self._points]

        return np.argsort(distances)[:self._k]

    @staticmethod
    def dist(x1, x2):
        """returns Euclidean distance between x1 and x2"""
        return np.linalg.norm(x1 - x2)



class ClassificationKNN(KNN):
    def __init__(self, k):
        """object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """predict labels for X_test and return predicted labels """
        X_test = self.get_scaler().transform(X_test)
        closest_neighbors = np.zeros(shape=(X_test.shape[0], self._k))
        for i in range(X_test.shape[0]):
            closest_neighbors[i] = np.array(KNN.neighbours_indices(self, X_test[i]))

        closest_labels = closest_neighbors
        for i in range(closest_labels.shape[0]):
            for j in range(closest_labels.shape[1]):
                # now we get for each point the labels of the closest k points to it
                closest_labels[i][j] = self._labels[int(closest_neighbors[i][j])]

        result = [self.predect_for_one(closest_labels, t) for t in range(closest_labels.shape[0])]

        return result

    def predect_for_one(self, closest_labels, t):
        temp = scipy.stats.mode(closest_labels[t])
        return temp[0][0]


class RegressionKNN(KNN):
    def __init__(self, k):
        """object instantiation, parent class instantiation """
        super().__init__(k)

    def predict(self, X_test):
        """predict labels for X_test and return predicted labels """
        X_test = self.get_scaler().transform(X_test)
        predicted_labels = [self.predict_point(x_test) for x_test in X_test]
        return predicted_labels

    def predict_point(self, x):
        """ predict label for a given point x and return predicted label """
        k_nearest_labels = [(self._points[i]) for i in self.neighbours_indices(x)]
        predicted = np.mean(np.array(k_nearest_labels))

        return predicted

        """if len(self._points) == 0:
            print('Please train the model first')
            return []
        if type(X_test) != list:
            # In case single point is provided
            X_test = [X_test]
        result = [self._labels[i] for i in KNN.neighbours_indices(self, X_test)]
        return np.mean(result)"""
