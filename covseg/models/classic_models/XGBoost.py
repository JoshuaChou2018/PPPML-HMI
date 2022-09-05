import xgboost as xgb
import xgbfir
import Tool_Functions.load_in_data as load_in_data
import Tool_Functions.Functions as Functions
import Tool_Functions.performance_metrics as metrics
import numpy as np
np.set_printoptions(suppress=True)
# keys: ['filename', 'target', 'feature_names', 'data']
# 'filename': '/home/zhoul0a/Desktop/example_files/data_prognosis.csv'
array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/analysis/healthy_mean_std/positive_negative_relations.npy')
# array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/analysis/healthy_mean_std/optimal_dice_threshold_healthy_relations.npy')
feature_name_list = ["mean_certainty_health", "std_certainty_health"]
data_set_dict = {'filename': None, 'target': array[:, 2], 'feature_names': feature_name_list, 'data': array[:, 0: 2]}
feature_list = ["mean_certainty_health", "std_certainty_health", "balance_certainty"]
data_array = np.array(array)
data_path = None
total_lines = np.shape(array)[0]


def delete_rows(original_array, row_list):
    shape = np.shape(original_array)
    return_array = np.zeros((shape[0] - len(row_list), shape[1]), 'float32')
    pointer = 0
    for i in range(shape[0]):
        if i in row_list:
            continue
        return_array[pointer, :] = original_array[i, :]
        pointer += 1
    return return_array


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

    r_score, p = metrics.pearson_correlation_coefficient(predicted, gt)
    root_mean_error = metrics.norm_mean_error(predicted, gt, 2)
    abs_mean_error = metrics.norm_mean_error(predicted, gt, 1)

    if show:
        print("pearson score:", r_score, "p:", p)
        r_spearman, p_spearman = metrics.spearman_ranking_correlation_coefficient(predicted, gt)
        print("spearman score:", r_spearman, "p_spearman:", p_spearman)
        print("root mean error:", root_mean_error)
        print("abs mean error:", abs_mean_error)
        Functions.show_data_points(gt, predicted, x_name='Ground Truth', y_name='Predicted Value', title='Prediction of Tumor Certainty', data_label=None,
                                   save_path='/home/zhoul0a/Desktop/Predict_SGRQ')

    if not difference:
        return model, r_score, p, root_mean_error, abs_mean_error
    difference_abs = np.abs(np.array(gt) - np.array(predicted))
    return model, r_score, p, root_mean_error, abs_mean_error, difference_abs


improve = []
delete_list_already = [68, 106, 110, 126] + [44, 86, 122] + [6, 50, 102] + [17, 22, 40, 42, 81]
print(np.shape(array)[0] - len(delete_list_already))
data_array = delete_rows(array, delete_list_already)
xg_regression(2, [0, 1], show=True)
exit()
temp_list = list(delete_list_already)
for value in delete_list_already:
    print('\n')
    print(value)
    temp_list.remove(value)
    data_array = delete_rows(array, temp_list)
    model, r_score, p, root_mean_error, abs_mean_error = xg_regression(2, [0, 1], show=False)
    print(r_score - 0.7550571224972771)
    temp_list = list(delete_list_already)
exit()
exit()
exit()
for line in range(total_lines):
    if line in delete_list_already:
        improve.append(0)
        continue
    delete_line = list(delete_list_already)
    delete_line.append(line)
    data_array = delete_rows(array, delete_line)
    model, r_score, p, root_mean_error, abs_mean_error = xg_regression(2, [0, 1], show=False)
    improve.append(r_score - 0.8193537445651855)
    print(line, r_score - 0.8193537445651855)

Functions.pickle_save_object('/home/zhoul0a/Desktop/improve_0.pickle', improve)
print(improve)
exit()
exit()


exit()






# highest_delete = [7, 36, 49, 68, 102, 106, 110, 116, 119, 122, 126, 133, 146, 150]
# [49, 68, 106, 110, 126] + [0, 4, 14, 15, 17, 28, 33, 36, 38, 34, 42, 43, 44, 82, 86, 92, 105, 111, 119, 122, 131, 136, 158]
# [49, 68, 106, 110, 126] + [17, 44, 86, 111, 122] + [6, 23, 42, 43, 50, 76, 88, 102, 130, 131, 145, 147, 154]
# [68, 106, 110, 126] + [44, 86, 122] + [6, 50, 102] + [17, 22, 40, 42, 81]
# [68, 106, 110, 126] + [44, 86, 122] + [6, 50, 102] + [17, 22, 40, 42, 81] + [49, 88, 111]



#  hayida: PCC 0.75: delete [68, 106, 110, 126] + [44, 86, 122] + [6, 50, 102] + [17, 22, 40, 42, 81]
#  the fn of the line use: fn_list = os.listdir(dict_for_probability_map)[line]
#  array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/analysis/healthy_mean_std/positive_negative_relations.npy')


"""
generate zheda PCC = 0.8738
delete_list_already = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 33, 39, 42,
                       44, 49, 54, 58, 60, 66, 74, 75, 76, 77, 78, 80, 85, 87, 88, 90,
                       96, 99, 102, 104, 107, 110, 111, 116, 120, 121, 124, 126, 128, 131,
                       133, 134, 145, 146, 148, 149, 153, 154, 161] + [31, 34, 36, 40, 45, 55, 61, 68, 79, 113, 114, 119, 127, 140, 155, 156]
array = np.load('/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/analysis/healthy_mean_std/optimal_dice_threshold_healthy_relations.npy')

"""