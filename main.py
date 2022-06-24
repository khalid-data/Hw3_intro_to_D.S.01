# This is a sample Python script.
import pandas as pd
import data
import knn
import cross_validation
import evaluation


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    #load data from csv file
    unwanted_features = ['timestamp', 'cnt', 'weather_code', 'is_holiday', 'is_weekend']
    df = data.load_data('london_sample_2500.csv')
    df = df.drop(unwanted_features, axis=1)

    folds = data.get_folds()
    y = df['season'].to_numpy()
    y = data.adjust_labels(y)
    X = df.drop(['season'], axis=1).to_numpy()

    X = data.add_noise(X)

    k_list = [3, 5, 11, 25, 51, 75, 101]

    #part 1
    print('Part 1 – Classification')

    mean_list, std_list = cross_validation.model_selection_cross_validation(knn.ClassificationKNN, k_list, X
                                                                            , y, folds, evaluation.f1_score)
    for i in range(len(mean_list)):
        print("k=" + str(k_list[i]) + ",mean score:" + "{:.4f}".format(mean_list[i])
              + ",std of scores:" + "{:.4f}".format(std_list[i]))

if __name__ == '__main__':
    main()


