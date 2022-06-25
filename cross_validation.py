import numpy as np
import data


def cross_validation_score(model, X, y, folds, metric):
    """run cross validation on X and y with specific model by given folds. Evaluate by given metric. """
    list = []

    for train_indices, validation_indices in folds.split(X):
        train_set = X[train_indices]
        train_label = [y[i] for i in train_indices]
        model.fit(train_set, train_label)  # check what parameters model takes

        test_set = X[validation_indices]
        real_label = [y[i] for i in validation_indices]

        """scaler = model.get_scaler()
        test_set = scaler.transform(test_set)"""
        y_pred = model.predict(test_set)

        list.append(metric(real_label, y_pred))

    return list


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    mean_list = []
    std_list = []
    for k in k_list:
        metric_score = np.array(cross_validation_score(model(k), X, y, folds, metric))
        mean_list.append(np.mean(metric_score))
        std_list.append(np.std(metric_score, ddof=1))

    return mean_list, std_list
