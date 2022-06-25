# This is a sample Python script.
import pandas as pd
import data
import numpy as np
import knn
import cross_validation
import evaluation


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    #load data from csv file
    df = data.load_data('london_sample_2500.csv')
    folds = data.get_folds()
    k_list = [3, 5, 11, 25, 51, 75, 101]

    wanted_features = ['t1', 't2', 'wind_speed', 'hum']

    y = data.adjust_labels(df['season'].to_numpy())
    X = data.add_noise(df[wanted_features].to_numpy())


    #part 1
    print('Part 1 – Classification')
    model = knn.ClassificationKNN
    metric = evaluation.f1_score
    mean_list, std_list = cross_validation.model_selection_cross_validation(model, k_list, X, y, folds, metric)
    for i in range(len(mean_list)):
        print("k=" + str(k_list[i]) + ", mean score: " + "{:.4f}".format(mean_list[i])
              + ", std of scores: " + "{:.4f}".format(std_list[i]))
    evaluation.visualize_results(k_list, mean_list, 'f1_scores', 'Classification', 'plot1.png')

    #part 2
    print()
    print('Part2 - Regression')
    wanted_features_R = ['t1', 't2', 'wind_speed']
    model_R = knn.RegressionKNN
    metric_R = evaluation.rmse
    temp = df[wanted_features_R]
    x1 = data.add_noise(np.array(temp))
    x_R = data.StandardScaler().fit_transform(x1)
    y_R = df['hum'].to_numpy()
    mean_list_R, std_list_R = cross_validation.model_selection_cross_validation(model_R, k_list, x_R, y_R,
                                                                                folds, metric_R)
    for i in range(len(mean_list_R)):
        print("k=" + str(k_list[i]) + ", mean score: " + "{:.4f}".format(mean_list_R[i])
              + ", std of scores: " + "{:.4f}".format(std_list_R[i]))
    evaluation.visualize_results(k_list, mean_list_R, 'RMSE', 'Regression', 'plot2.png')

if __name__ == '__main__':
    main()


