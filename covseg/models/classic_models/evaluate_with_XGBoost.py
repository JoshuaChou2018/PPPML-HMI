"""
for XGBoost prediction, feature selection and ranking.
standard data format, two global variables:
feature_list: is freq list of names: ['SGRQ', 'invisible_lesion', ...]
data_array: freq array with shape (number_samples, len(feature_list))

xg_regression(target_column, data_column_list, test_num=1, evaluate=True, save_report_path=None, show=True)
e.g.
xg_regression(1, [2, 5, 7]) means using column [2, 5, 7] to predict column 1

select_features(target_column, candidate_feature_list=None, importance_rank=None, baseline_r=None, depth=0,
                    leave_count=6)
e.g.
select_features(0, candidate_feature_list=np.arrange(1, 50)) means select features from 1-49 column for predcit column 0
"""

import xgboost as xgb
import xgbfir
import Tool_Functions.load_in_data as load_in_data
import Tool_Functions.performance_metrics as metrics
import Tool_Functions.Functions as Functions
import numpy as np
import random

np.set_printoptions(suppress=True)


def delete_rows(array, row_list):
    shape = np.shape(array)
    return_array = np.zeros((shape[0] - len(row_list), shape[1]), 'float32')
    pointer = 0
    for i in range(shape[0]):
        if i in row_list:
            print("delete row:", array[i, 0: 5])
            continue
        return_array[pointer, :] = array[i, :]
        pointer += 1
    return return_array


def shuffle_rows(array):
    rows = np.shape(array)[0]
    index = list(np.arange(0, rows))
    random.shuffle(index)
    return array[index]


prognosis = True
if prognosis:
    data_path = '/home/zhoul0a/Desktop/Lung_CAD_NMI/source_codes/data_linkage/data_folder/final_data.csv'
    data_set_dict = load_in_data.load_in_csv(0, data_path)
    # the first column is the patient id
    data_array = np.array(data_set_dict['data'], 'float32')
    feature_list = data_set_dict['feature_names']
    data_array = delete_rows(data_array, [19, 29, 33, 45, 22, 34])
    data_array = shuffle_rows(data_array)
else:
    data_path = None
    # data_set = Functions.pickle_load_object('/home/zhoul0a/Desktop/prognosis_project/feature_abstracted/oxy_index_dataset_interpolate.pickle')
    data_set = Functions.pickle_load_object('/home/zhoul0a/Desktop/data_access/oxy_index_dataset_interpolate.pickle')
    data_array = data_set['sample_array']

    print(data_array[:, 0])
    # data_array = delete_rows(data_array, [31, 55, 64, 70, 4, 39, 40])
    print(np.shape(data_array))
    feature_list = data_set['feature_list']
    print(feature_list)
    print(list(data_set.keys()))


def xg_regression(target_column, data_column_list, test_num=1, evaluate=True, save_report_path=None, show=True,
                  difference=False):
    """
    :param difference: whether return the difference of predicted and ground truth
    :param test_num: how many samples are used as test set
    :param save_report_path: if None, do not save performance report.
    :param target_column: freq int, which column of the data array are considered as gt?
    :param data_column_list: freq tuple of int, which column are used as data?
    :param evaluate: report the performance of the model
    :param show: whether print out the intermediate results
    :return: the model
    """
    feature_names = []
    for column in data_column_list:
        feature_names.append(feature_list[column])
    rows = np.shape(data_array)[0]
    columns = len(data_column_list)
    data = np.zeros((rows, columns), 'float32')
    pointer = 0
    for column in data_column_list:
        data[:, pointer] = data_array[:, column]
        pointer += 1
    data_set = {'filename': data_path, 'target': data_array[:, target_column], 'feature_names': feature_names,
                'data': data}
    if show:
        print("there are:", rows, "samples and:", columns, "features")
        print("the target feature being predict is:", feature_list[target_column])
        print("the ground truth is:", data_array[:, target_column])
        print("the features are:", feature_names)
        print("the data for the first row:")
        print(data[0, :])

    model = xgb.XGBRegressor().fit(data_set['data'], data_set['target'])
    if not evaluate:
        if save_report_path is not None:
            xgbfir.saveXgbFI(model, feature_names=data_set['feature_names'], OutputXlsxFile=save_report_path)
        print(model.feature_importances_)
        return model

    predicted = []
    gt = data_array[:, target_column]

    temp_data = np.zeros((rows - test_num, columns), 'float32')
    temp_target = np.zeros((rows - test_num,), 'float32')

    for test_start in range(0, rows, test_num):

        if test_start + test_num < rows:
            temp_data[0: test_start, :] = data[0: test_start, :]
            temp_target[0: test_start] = gt[0: test_start]
            test_num_this_fold = test_num
            temp_data[test_start::, :] = data[test_start + test_num::, :]
            temp_target[test_start::] = gt[test_start + test_num::]
        else:
            temp_data[:, :] = data[0: rows - test_num, :]
            temp_target = gt[0: rows - test_num]
            test_num_this_fold = rows - test_start

        temp_model = xgb.XGBRegressor().fit(temp_data, temp_target)
        predicted = predicted + list(temp_model.predict(data[test_start: test_start + test_num_this_fold, :]))

    new_gt = []
    new_pred = []
    for i in range(len(gt)):
        if gt[i] > 390:  # 400 is normal people, which is outlier
            continue
        new_gt.append(gt[i])
        new_pred.append(predicted[i])
    gt = new_gt
    predicted = new_pred

    r_score, p = metrics.pearson_correlation_coefficient(predicted, gt)
    root_mean_error = metrics.norm_mean_error(predicted, gt, 2)
    abs_mean_error = metrics.norm_mean_error(predicted, gt, 1)

    if show:
        print("pearson score:", r_score, "p:", p)
        r_spearman, p_spearman = metrics.spearman_ranking_correlation_coefficient(predicted, gt)
        print("spearman score:", r_spearman, "p_spearman:", p_spearman)
        print("root mean error:", root_mean_error)
        print("abs mean error:", abs_mean_error)
        Functions.show_data_points(gt, predicted, x_name='Ground Truth', y_name='Predicted Value', title='Prediction of Oxygenation Index', data_label=None,
                                   save_path='/home/zhoul0a/Desktop/Predict_SGRQ')
    if not difference:
        return model, r_score, p, root_mean_error, abs_mean_error
    difference_abs = np.abs(np.array(gt) - np.array(predicted))
    return model, r_score, p, root_mean_error, abs_mean_error, difference_abs


def select_features(target_column, candidate_feature_list=None, importance_rank=None, baseline_r=None, depth=0,
                    leave_count=6):
    """
    this functions will find optimal feature combinations for xg_regression
    :param leave_count: leave 'leave_count' out during validation
    :param depth: too track the recursive depth
    :param baseline_r: r_score using all candidate features
    :param importance_rank: list of int, from least important to most important
    :param candidate_feature_list: list of int, find optimal feature combinations from candidate_features
    :param target_column: int, the column we want to predict
    :return: list of int, refer to column numbers, which is the optimal features for xg_regression to predict the target
    """
    if candidate_feature_list is not None:
        assert target_column not in candidate_feature_list
    if candidate_feature_list is None:
        candidate_feature_list = list(np.arange(0, 89))
        candidate_feature_list.remove(target_column)
    # random.shuffle(candidate_feature_list[int(len(candidate_feature_list) / 2)::])
    current_feature_combination = candidate_feature_list
    first_time = False
    if baseline_r is None:
        first_time = True
    if importance_rank or baseline_r is None:
        model, baseline_r, _, _, _ = xg_regression(target_column, current_feature_combination, leave_count, show=False)
        importance_list = list(zip(list(model.feature_importances_), current_feature_combination))
        importance_list.sort()
        importance_rank = []
        for item in importance_list:
            importance_rank.append(item[1])
    print("baseline r:", baseline_r)

    if first_time:
        for i in range(10):
            importance_rank.reverse()
            model, baseline_r, _, _, _ = xg_regression(target_column, importance_rank, leave_count, show=False)
            importance_list = list(zip(list(model.feature_importances_), current_feature_combination))
            importance_list.sort()
            importance_rank = []
            for item in importance_list:
                importance_rank.append(item[1])
        current_feature_combination = importance_rank
        current_feature_combination.reverse()

    optimal_feature_combination = list(current_feature_combination)
    for feature in importance_rank:
        current_feature_combination.remove(feature)
        # random.shuffle(current_feature_combination)
        model, current_r, _, _, _ = xg_regression(target_column, current_feature_combination, leave_count, show=False)
        if current_r > baseline_r:  # delete this feature improved the performance

            new_importance_list = list(zip(list(model.feature_importances_), current_feature_combination))
            new_importance_list.sort()
            new_importance_rank = []
            new_combination = []
            for item in new_importance_list:
                new_importance_rank.append(item[1])
                new_combination.append(item[0])
            new_combination.reverse()
            return select_features(
                target_column, current_feature_combination, new_importance_rank, current_r, depth=depth + 1)

        current_feature_combination = list(optimal_feature_combination)
    # this means delete any feature will decrease the performance
    model, optimal_r, _, _, _ = xg_regression(target_column, optimal_feature_combination, 1, show=False)
    importance_list_final = list(zip(list(model.feature_importances_), optimal_feature_combination))
    importance_list_final.sort()
    importance_rank_feature = []
    for item in importance_list_final:
        importance_rank_feature.append(item[1])
    importance_rank_feature.reverse()
    best_combination_names = []
    for feature in importance_rank_feature:
        best_combination_names.append(feature_list[feature])
    if optimal_r < baseline_r:
        best_combination_names = []
        for feature in importance_rank:
            best_combination_names.append(feature_list[feature])
        return baseline_r, importance_rank, best_combination_names
    return optimal_r, importance_rank_feature, best_combination_names


'''
column 0, 1 are subjective sequela,
column 2 - 15 are objective sequela,
column 16 - 25 are CT sequela,

column 26 - 30 are basic information like sex, age, BMI

column 31 - 32 are hospitalization CT features,
column 33 - 36 are hospitalization clinical features recommended by doctors,

'''

if __name__ == '__main__':

    if prognosis:

        # model, current_r, _, _, _, differ_1 = xg_regression(1, [17, 16, 23], difference=True)
        model, current_r, _, _, _, differ_1 = xg_regression(1, [46, 58], difference=True)
        print(model.feature_importances_)
        exit()

        # model, current_r, _, _, _, differ_2 = xg_regression(1, [10, 4, 2, 31], difference=True)  # -R1

        model, current_r, _, _, _, differ_2 = xg_regression(1, [31, 4], difference=True)  # -R2  (can further remove 31)

        more_predict = differ_2 - differ_1

        print(more_predict)

        print(np.mean(more_predict) / np.std(more_predict) * np.sqrt(46))

        exit()

        candidate = list(np.arange(2, 36))
        candidate.remove(16)
        best_features = select_features(1, candidate_feature_list=[18, 31, 4, 10, 17])  # [0, 1, 17, 8, 30, 3, 13, 18, 9, 11, 27, 16, 19, 4, 26, 20, 31], leave_count=1)
        print(best_features)
        exit()
        model, current_r, _, _, _, differ_2 = xg_regression(1, [39, 6, 31, 79, 30, 27, 10], difference=True)

        more_predict = differ_2 - differ_1

        print(more_predict)

        print(np.mean(more_predict) / np.std(more_predict) * np.sqrt(46))
        exit()
    else:

        model, _, _, _, _, differ_1 = xg_regression(0, [5, 4, 10, 7, 3, 2], difference=True)
        print(model.feature_importances_)

        model, current_r, _, mae, _, differ_2 = xg_regression(0, [5, 4, 10, 6, 3, 2], difference=True)

        more_predict = differ_1 - differ_2

        more_lesion = data_array[:, 7] - data_array[:, 6]
        print(len(more_predict), len(more_lesion))

        print(metrics.pearson_correlation_coefficient(more_predict, more_lesion))

        print(more_predict)

        print(np.mean(more_predict) / np.std(more_predict) * np.sqrt(71))
        exit()
        xg_regression(0, [2, 3, 4, 5, 6, 7, 8, 9, 10])
        # xg_regression(0, [1, 2, 3, 4, 5, 7, 8, 9, 10])
    exit()
    best_features = select_features(2, candidate_feature_list=[17, 8, 30, 3, 13, 18, 9, 11, 27, 16, 19, 4, 26, 20,
                                                               31])  # [0, 1, 17, 8, 30, 3, 13, 18, 9, 11, 27, 16, 19, 4, 26, 20, 31], leave_count=1)
    print(best_features)
    exit()
    model, current_r, _, _, _, differ_1 = xg_regression(1, [30, 31, 2], difference=True)
    model, current_r, _, _, _, differ_2 = xg_regression(1, [17, 16, 23], difference=True)
    print(np.mean(differ_1))
    print(np.std(differ_1) / np.sqrt(41))
    print(np.mean(differ_2))
    print(np.std(differ_2) / np.sqrt(41))
    print((np.mean(differ_1) - np.mean(differ_2)) / (2 / (1 / np.std(differ_1) + 1 / np.std(differ_2))) * np.sqrt(46))
    exit()
    print(feature_list)
    feature = 8
    print(feature_list[feature])
    # print(select_features(feature, candidate_feature_list=np.arange(9, 36)))
    model, current_r, _, _, _ = xg_regression(feature, [23, 19, 33, 12, 10, 9])
    print(model.feature_importances_)
    exit()






    save_path = '/home/zhoul0a/Desktop/prognosis_project/reports/xg_model_reports/objective_to_SGRQ.xlsx'

    best_features = select_features(1, candidate_feature_list=[17, 16, 23], leave_count=1)
    print(best_features)
    exit()


"""


feature:  best combination:             
1         [16, 17, 18, 23, 26]  
pearson score: 0.630728401018637 p: 7.558037543760469e-06
root mean error: 8.412258281284863
abs mean error: 6.537071568625314
1         [17, 16, 23, 18, 26]
pearson score: 0.6435305283989399 p: 4.285679972798166e-06
root mean error: 8.39743499399782
abs mean error: 6.638669439724514

1         [17, 16, 23]
pearson score: 0.6765540642442747 p: 8.724834929469879e-07
spearman score: 0.6806604918776398 p_spearman: 7.057293927735232e-07
root mean error: 7.968295789576193
abs mean error: 6.484878994169689
"""
