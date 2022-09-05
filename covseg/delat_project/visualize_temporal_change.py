import numpy as np
import matplotlib.pyplot as plt

from csaps import csaps
import os
import Tool_Functions.Functions as Functions

from delat_project.data_linkage.delta_patient_information import calendar_time_difference

dataset = Functions.pickle_load_object('/home/zhoul0a/Desktop/COVID-19 delta/dataset/dataset.pickle')
save_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/visualization/temporal_changes/'


def impute_and_visualize(patient_id, radiomic):
    scan_date = list(dataset[patient_id][radiomic].keys())
    passed_date_20200101 = []

    radiomic_list = []

    for time in scan_date:
        passed_date_20200101.append(calendar_time_difference('20200101', time))  # x
        radiomic_list.append(dataset[patient_id][radiomic][time])  # y
    passed_date_20200101 = np.array(passed_date_20200101, 'float32')
    initial_date_passed = np.min(passed_date_20200101)
    passed_date_20200101 = passed_date_20200101 - initial_date_passed
    inpatient_date = np.max(passed_date_20200101)

    x = passed_date_20200101
    y = radiomic_list

    length = len(x)

    temp_list = []
    for i in range(length):
        temp_list.append((x[i], y[i]))

    def compare(item_a, item_b):
        if item_a[0] > item_b[0]:
            return 1
        return -1

    temp_list = Functions.customized_sort(temp_list, compare)
    x = []
    y = []
    for i in range(length):
        x.append(temp_list[i][0])
        y.append(temp_list[i][1])

    xs = np.arange(0, inpatient_date + 1)

    ys = csaps(x, y, xs, smooth=0.99)

    #plt.plot(x, y, 'o', xs, ys, '-')
    plt.plot(x, y, 'o', label='Observed value')
    plt.plot(xs, ys, '-', label='Simulated value')
    plt.xlabel('Days after the first scan')
    plt.ylabel('Lesion ratio')
    plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
    plt.title('Lesion Temporal Change')
    plt.show()
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    plt.savefig('/home/zhoul0a/Desktop/1.png')
    plt.close()
    plt.show()

    exit()

    save_dict = save_top_dict + radiomic + '/'
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    severity = dataset[patient_id]['clinical_severity']
    save_path = save_dict + patient_id + '_' + str(severity)
    plt.savefig(save_path)
    plt.close()


# impute_and_visualize('P0140264', 'lesion_ratio')
# exit()


def half_decay_and_double_time(patient_id, radiomic):
    scan_date = list(dataset[patient_id][radiomic].keys())
    passed_date_20200101 = []

    radiomic_list = []

    for time in scan_date:
        passed_date_20200101.append(calendar_time_difference('20200101', time))  # x
        radiomic_list.append(dataset[patient_id][radiomic][time])  # y
    passed_date_20200101 = np.array(passed_date_20200101, 'float32')
    initial_date_passed = np.min(passed_date_20200101)
    passed_date_20200101 = passed_date_20200101 - initial_date_passed
    inpatient_date = np.max(passed_date_20200101)

    x = passed_date_20200101
    y = radiomic_list

    length = len(x)

    temp_list = []
    for i in range(length):
        temp_list.append((x[i], y[i]))

    def compare(item_a, item_b):
        if item_a[0] > item_b[0]:
            return 1
        return -1

    temp_list = Functions.customized_sort(temp_list, compare)
    x = []
    y = []
    for i in range(length):
        x.append(temp_list[i][0])
        y.append(temp_list[i][1])

    xs = np.arange(0, inpatient_date + 2)

    ys = csaps(x, y, xs, smooth=0.99)

    max_x = 0
    max_y = -1000000

    for current_x in range(0, int(inpatient_date) + 1):
        if max_y < ys[current_x]:
            max_y = ys[current_x]
            max_x = current_x

    half_y = max_y / 2

    previous_half_x = -1000
    later_half_x = -1000

    for current_x in range(max_x, -1, -1):
        if ys[current_x] <= half_y:

            one_day_later = ys[current_x + 1]

            previous_half_x = current_x + (half_y - ys[current_x]) / (one_day_later - ys[current_x])
            break
    for current_x in range(max_x, int(inpatient_date) + 1, 1):
        if ys[current_x] <= half_y:

            one_day_previous = ys[current_x - 1]

            later_half_x = current_x - (half_y - ys[current_x]) / (one_day_previous - ys[current_x])
            break
    double_time = -1000
    decay_time = -1000

    if previous_half_x >= 0:
        double_time = max_x - previous_half_x
    if later_half_x >= 0:
        decay_time = later_half_x - max_x

    return double_time, decay_time


def compare_progression_pattern():
    name_list = os.listdir(
        '/home/zhoul0a/Desktop/COVID-19 delta/visualization/temporal_changes/lesion_severity_enhance/')

    half_decay_previous = []
    half_decay_delta = []
    double_time_previous = []
    double_time_delta = []
    for pic_name in name_list:
        patient = pic_name.split('_')[0]
        spread_interval, decay_interval = half_decay_and_double_time(patient, 'lesion_severity_ct')

        if dataset[patient]['strain'] == 'delta':
            if spread_interval > 0:
                double_time_delta.append(spread_interval)
            if decay_interval > 0:
                half_decay_delta.append(decay_interval)
        if dataset[patient]['strain'] == 'baseline':
            if spread_interval > 0:
                double_time_previous.append(spread_interval)
            if decay_interval > 0:
                half_decay_previous.append(decay_interval)

    half_decay_previous.sort()
    half_decay_delta.sort()
    double_time_previous.sort()
    double_time_delta.sort()
    print(half_decay_previous)
    print(half_decay_delta)
    print(double_time_previous)
    print(double_time_delta)

    print(np.average(half_decay_previous), np.std(half_decay_previous))
    print(np.average(half_decay_delta), np.std(half_decay_delta))
    print(np.average(double_time_previous), np.std(double_time_previous))
    print(np.average(double_time_delta), np.std(double_time_delta))


def follow_up_recovered_rate(patient_id, radiomic):
    scan_date = list(dataset[patient_id][radiomic].keys())
    passed_date_20200101 = []

    radiomic_list = []

    for time in scan_date:
        passed_date_20200101.append(calendar_time_difference('20200101', time))  # x
        radiomic_list.append(dataset[patient_id][radiomic][time])  # y
    passed_date_20200101 = np.array(passed_date_20200101, 'float32')
    initial_date_passed = np.min(passed_date_20200101)
    passed_date_20200101 = passed_date_20200101 - initial_date_passed
    inpatient_date = np.max(passed_date_20200101)

    x = passed_date_20200101
    y = radiomic_list

    length = len(x)

    temp_list = []
    for i in range(length):
        temp_list.append((x[i], y[i]))

    def compare(item_a, item_b):
        if item_a[0] > item_b[0]:
            return 1
        return -1

    temp_list = Functions.customized_sort(temp_list, compare)
    x = []
    y = []
    for i in range(length):
        x.append(temp_list[i][0])
        y.append(temp_list[i][1])

    xs = np.arange(0, inpatient_date + 2)

    ys = csaps(x, y, xs, smooth=0.99)

    max_x = 0
    max_y = -1000000

    for current_x in range(0, int(inpatient_date) + 1):
        if max_y < ys[current_x]:
            max_y = ys[current_x]
            max_x = current_x

    follow_up = ys[int(inpatient_date)]
    if max_x > follow_up > 0:
        return follow_up / max_y

    return None


def compare_follow_up():
    patient_id_list = list(dataset.keys())
    patient_id_list.remove('radiomics_processed_list')
    patient_id_list.remove('spatial_processed_list')

    recover_rate_previous = []
    recover_rate_delta = []
    for patient in patient_id_list:
        # patient = patient.split('_')[0]
        try:
            rate = follow_up_recovered_rate(patient, 'lesion_ratio')
            if rate == 1:
                continue
            if rate is None:
                continue
            if dataset[patient]['clinical_severity'] < 1:
                continue

            if dataset[patient]['strain'] == 'delta':
                recover_rate_delta.append(rate)
            if dataset[patient]['strain'] == 'baseline':
                recover_rate_previous.append(rate)
        except:
            continue

    recover_rate_delta.sort()
    recover_rate_previous.sort()
    print(recover_rate_previous)
    print(recover_rate_delta)
    print(np.average(recover_rate_previous), np.std(recover_rate_previous))
    print(np.average(recover_rate_delta), np.std(recover_rate_delta))


def abstract_distribution_at_max_lesion(patient_id, radiomic='lesion_ratio'):
    if radiomic + '_distribution' in list(dataset[patient_id].keys()):
        distribution_time = list(dataset[patient_id][radiomic + '_distribution'].keys())
    else:
        return None
    highest_value = -1000
    highest_time = distribution_time[0]
    for scan_time in distribution_time:
        current_value = dataset[patient_id][radiomic][scan_time]
        if current_value > highest_value:
            highest_value = current_value
            highest_time = scan_time
    distribution = np.array(dataset[patient_id][radiomic + '_distribution'][highest_time], 'float32')

    return distribution / np.sum(distribution)


def abstract_age_distribution():
    patient_id_list = list(dataset.keys())
    patient_id_list.remove('radiomics_processed_list')
    patient_id_list.remove('spatial_processed_list')
    age_delta = []
    age_baseline = []
    for patient in patient_id_list:
        if dataset[patient]['strain'] == 'delta':
            age_delta.append(dataset[patient]['age'])
        if dataset[patient]['strain'] == 'baseline':
            age_baseline.append(dataset[patient]['age'])
    print(age_delta)
    print(age_baseline)


abstract_age_distribution()
exit()


def lesion_distribution_for_dataset():
    patient_id_list = list(dataset.keys())
    patient_id_list.remove('radiomics_processed_list')
    patient_id_list.remove('spatial_processed_list')
    distribution_info = []
    for patient_id in patient_id_list:
        strain = dataset[patient_id]['strain']
        current_distribution = abstract_distribution_at_max_lesion(patient_id)
        if current_distribution is not None:
            distribution_info.append((strain, current_distribution))
    return distribution_info


def imputation_test_for_spatial_distribution(test_count):
    distribution_info = lesion_distribution_for_dataset()
    distribution_delta = []
    delta_case = 0
    distribution_baseline = []
    baseline_case = 0
    for item in distribution_info:
        if item[0] == 'delta':
            delta_case += 1
            distribution_delta.append(item[1])
        if item[0] == 'baseline':
            baseline_case += 1
            distribution_baseline.append(item[1])

    distribution_delta = np.sum(distribution_delta, axis=0) / delta_case
    distribution_baseline = np.sum(distribution_baseline, axis=0) / baseline_case

    print(distribution_delta)
    print(distribution_baseline)
    exit()

    def difference(distribution_a, distribution_b):
        return np.sum(np.abs(distribution_a - distribution_b))

    all_distribution = []
    for item in distribution_info:
        all_distribution.append(item[1])

    half_item = int(len(all_distribution) / 2)
    other_half = len(all_distribution) - half_item

    delta_baseline_difference = difference(distribution_delta, distribution_baseline)

    import random

    more_difference_count = 0
    for i in range(test_count):
        random.shuffle(all_distribution)
        split_left, split_right = all_distribution[0: half_item], all_distribution[half_item::]
        distribution_left = np.sum(split_left, axis=0) / half_item
        distribution_right = np.sum(split_right, axis=0) / other_half
        current_difference = difference(distribution_left, distribution_right)
        if current_difference > delta_baseline_difference:
            more_difference_count += 1
        if i % 10000 == 0:
            print(more_difference_count / (i + 1))
    print('final', more_difference_count / test_count)


# imputation_test_for_spatial_distribution(test_count=1000000)
