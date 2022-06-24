import numpy as np
import data


def cross_validation_score(model, X, y, folds, metric):
    """run cross validation on X and y with specific model by given folds. Evaluate by given metric. """
    list = []

    for train_indices, validation_indices in folds.split(X):
        train_set = X[train_indices]
        train_label = [y[i] for i in train_indices]
        test_set = X[validation_indices]
        test_label = [y[i] for i in validation_indices]

        model.fit(train_set, train_label)  # check what parameters model takes
        scaler = model.get_scaler()
        test_set = scaler.transform(test_set)
        y_pred = model.predict(test_set)
        """print('y_pred')
        print(y_pred)
        print('y_true')
        print(test_label)"""
        list.append(metric(test_label, y_pred))

    return list


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    mean_list = []
    std_list = []
    for k in k_list:
        temp = np.array(cross_validation_score(model(k), X, y, folds, metric))
        mean_list.append(np.mean(temp))
        std_list.append(np.std(temp))

    return mean_list, std_list
