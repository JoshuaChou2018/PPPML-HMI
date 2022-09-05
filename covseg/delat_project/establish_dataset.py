"""
dataset is freq dict with {patient_id: {measure_id: {time: value}}}
time is like '20210328' or 'static' for static value, like sex.

"""

import Tool_Functions.Functions as Functions
import Tool_Functions.load_in_data as loader
import numpy as np
import os

# completed, 0, 1, 2, 3, 4,
measure_id_name_dict = {
    0: 'strain',
    1: 'clinical_severity',
    2: 'vaccine',
    3: 'sex',
    4: 'age',
    5: 'height',
    6: 'weight',
    7: 'BMI',
    8: 'smoke',
    9: 'disease_history',
    10: 'additional_information',

    11: 'oxygen_index',
    12: 'lym_count',
    13: 'lactate',   # 11-13 are related to the treatment

    14: 'IL-6',
    15: 'C-reactive_protein',
    16: 'ferritin',   # 14-16 are inflammatory markers

    17: 'D-Dimer',  # related to coagulation

    18: 'time_to_pcr_negative',

    19: 'lung_volume',  # in ml
    20: 'lesion_volume',  # in ml
    21: 'lesion_ratio',
    22: 'total_lesion_severity',
    23: 'lesion_severity_ratio',
    24: 'median_lesion_severity',

    25: 'lung_volume_upper',
    26: 'lesion_volume_upper',
    27: 'lesion_ratio_upper',
    28: 'total_lesion_severity_upper',
    29: 'lesion_severity_ratio_upper',
    30: 'median_lesion_severity_upper',

    31: 'lung_volume_middle',
    32: 'lesion_volume_middle',
    33: 'lesion_ratio_middle',
    34: 'total_lesion_severity_middle',
    35: 'lesion_severity_ratio_middle',
    36: 'median_lesion_severity_middle',

    37: 'lung_volume_lower',
    38: 'lesion_volume_lower',
    39: 'lesion_ratio_lower',
    40: 'total_lesion_severity_lower',
    41: 'lesion_severity_ratio_lower',
    42: 'median_lesion_severity_lower',
}


def separate_time_and_value(time_value, year='2021'):
    """

    :param time_value: freq string with format month.date-value, e.g. 8.1-1.55
    :param year:
    :return: time, value       e.g. 20210801, 1.55
    """
    if not len(time_value.split('-')) == 2:
        print(time_value)
    t, v = time_value.split('-')

    if not len(t.split('.')) == 2:
        print(t)
    m, d = t.split('.')
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    return year + m + d, float(v)


def separate_several_time_and_value(data_flow):
    assert len(data_flow) > 4
    time_value = data_flow.split(' ')
    return_list = []
    for item in time_value:
        if len(item) < 3:
            continue
        return_list.append(separate_time_and_value(item))
    return return_list


# print(separate_several_time_and_value('8.15-283.47 8.17-275.41 8.19-321.95 8.24-343.13 9.18-239.6'))


if __name__ == '__main__':

    file_name_list = os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/branching_array/')
    dataset = Functions.pickle_load_object('/home/zhoul0a/Desktop/longxi_platform/delat_project/dataset.pickle')
    for name in file_name_list:
        patient_id, scan_time = name.split('_')
        if patient_id in ['P0140558', 'P0087135']:
            continue
        scan_time = scan_time[:-4]
        print(patient_id, scan_time)

        branching_array = np.load('/home/zhoul0a/Desktop/COVID-19 delta/branching_array/' + name)['array']
        lesion = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/lesions/' + name)['array']
        lung = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/lung_mask/' + name)['array']
        enhanced_array = np.load('/home/zhoul0a/Desktop/COVID-19 delta/enhanced_array/' + name)['array']

        upper = np.array(0.01 < branching_array, 'float32') * np.array(branching_array <= 4, 'float32')
        middle = np.array(4 < branching_array, 'float32') * np.array(branching_array <= 8, 'float32')
        lower = np.array(8 < branching_array, 'float32') * np.array(branching_array <= 15, 'float32')

        lung_volume = np.sum(lung) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_volume = np.sum(lesion) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_ratio = lesion_volume / lung_volume
        total_lesion_severity = np.sum(enhanced_array * lesion)
        lesion_severity_ratio = total_lesion_severity / lung_volume
        median_lesion_severity = total_lesion_severity / lesion_volume
        dataset[patient_id][19][scan_time] = lung_volume
        dataset[patient_id][20][scan_time] = lesion_volume
        dataset[patient_id][21][scan_time] = lesion_ratio
        dataset[patient_id][22][scan_time] = total_lesion_severity
        dataset[patient_id][23][scan_time] = lesion_severity_ratio
        dataset[patient_id][24][scan_time] = median_lesion_severity

        lung = upper
        lesion = upper * lesion
        lung_volume = np.sum(lung) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_volume = np.sum(lesion) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        total_lesion_severity = np.sum(enhanced_array * lesion)
        lesion_severity_ratio = total_lesion_severity / lung_volume
        median_lesion_severity = total_lesion_severity / lesion_volume
        dataset[patient_id][25][scan_time] = lung_volume
        dataset[patient_id][26][scan_time] = lesion_volume
        dataset[patient_id][27][scan_time] = lesion_ratio
        dataset[patient_id][28][scan_time] = total_lesion_severity
        dataset[patient_id][29][scan_time] = lesion_severity_ratio
        dataset[patient_id][30][scan_time] = median_lesion_severity

        lung = middle
        lesion = middle * lesion
        lung_volume = np.sum(lung) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_volume = np.sum(lesion) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_ratio = lesion_volume / lung_volume
        total_lesion_severity = np.sum(enhanced_array * lesion)
        lesion_severity_ratio = total_lesion_severity / lung_volume
        median_lesion_severity = total_lesion_severity / lesion_volume
        dataset[patient_id][31][scan_time] = lung_volume
        dataset[patient_id][32][scan_time] = lesion_volume
        dataset[patient_id][33][scan_time] = lesion_ratio
        dataset[patient_id][34][scan_time] = total_lesion_severity
        dataset[patient_id][35][scan_time] = lesion_severity_ratio
        dataset[patient_id][36][scan_time] = median_lesion_severity

        lung = lower
        lesion = lower * lesion
        lung_volume = np.sum(lung) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_volume = np.sum(lesion) * 334 / 512 * 334 / 512 / 1000 + 0.00000001
        lesion_ratio = lesion_volume / lung_volume
        total_lesion_severity = np.sum(enhanced_array * lesion)
        lesion_severity_ratio = total_lesion_severity / lung_volume
        median_lesion_severity = total_lesion_severity / lesion_volume
        dataset[patient_id][37][scan_time] = lung_volume
        dataset[patient_id][38][scan_time] = lesion_volume
        dataset[patient_id][39][scan_time] = lesion_ratio
        dataset[patient_id][40][scan_time] = total_lesion_severity
        dataset[patient_id][41][scan_time] = lesion_severity_ratio
        dataset[patient_id][42][scan_time] = median_lesion_severity

        print(dataset[patient_id])

    Functions.pickle_save_object('/home/zhoul0a/Desktop/longxi_platform/delat_project/dataset_1017.pickle', dataset)
    exit()
    data_array = loader.convert_csv_to_numpy(
        '/home/zhoul0a/Desktop/longxi_platform/delat_project/data_linkage/clinical_data.csv')
    print(data_array.shape)
    dataset = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/longxi_platform/delat_project/dataset.pickle')

    for i in range(np.shape(data_array)[0]):
        patient_id = data_array[i][0]
        if patient_id in ['P0140558', 'P0087135']:
            continue
        print(patient_id)
        # print(data_array[i][1])
        value = data_array[i][1]
        if not data_array[i][1] == '/':
            value = float(value)
        dataset[data_array[i][0]][5]['static'] = value  # height
        print(5)

        # print(data_array[i][2])
        value = data_array[i][2]
        if not data_array[i][2] == '/':
            value = float(value)
        dataset[data_array[i][0]][6]['static'] = value  # weight
        print(6)

        if not data_array[i][1] == '/' and not data_array[i][2] == '/':
            bmi = float(data_array[i][2]) / float(data_array[i][1]) / float(data_array[i][1])
            # print(bmi)
            dataset[data_array[i][0]][7]['static'] = bmi  # BMI
        else:
            dataset[data_array[i][0]][7]['static'] = '/'
        print(7)

        dataset[data_array[i][0]][8]['static'] = data_array[i][3]  # Smoke
        print(8)
        dataset[data_array[i][0]][9]['static'] = data_array[i][4]  # disease_history
        print(9)
        dataset[data_array[i][0]][18]['static'] = data_array[i][12]  # time_to_pcr_negative
        print(18)

        data_record = data_array[i][5]  # oxygen_index
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][11][sample[0]] = sample[1]
        print(11)

        data_record = data_array[i][6]  # lactate
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][13][sample[0]] = sample[1]
        print(13)

        data_record = data_array[i][7]  # lym_count
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][12][sample[0]] = sample[1]
        print(12)

        data_record = data_array[i][8]  # IL-6
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][14][sample[0]] = sample[1]
        print(14)

        data_record = data_array[i][9]  # CRP
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][15][sample[0]] = sample[1]
        print(15)

        data_record = data_array[i][10]  # ferritin
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][16][sample[0]] = sample[1]
        print(16)

        data_record = data_array[i][11]  # D-Dimer
        if not data_record == '/':
            sample_list = separate_several_time_and_value(data_record)
            for sample in sample_list:
                dataset[data_array[i][0]][17][sample[0]] = sample[1]
        print(17)

    Functions.pickle_save_object('/home/zhoul0a/Desktop/longxi_platform/delat_project/dataset.pickle', dataset)

    exit()
    temp_data = Functions.pickle_load_object('/home/zhoul0a/Desktop/COVID-19 delta/dataset/dataset.pickle')
    new_dataset = {}
    patient_id_list = list(temp_data.keys())[:-2]
    print(len(patient_id_list))
    print(patient_id_list)
    for patient_id in patient_id_list:
        new_dataset[patient_id] = {}
        for i in range(43):
            new_dataset[patient_id][i] = {}
    for patient_id in patient_id_list:
        print(patient_id)
        new_dataset[patient_id][0]['static'] = temp_data[patient_id]['strain']
        new_dataset[patient_id][1]['static'] = temp_data[patient_id]['clinical_severity']
        new_dataset[patient_id][2]['static'] = temp_data[patient_id]['vaccine']
        new_dataset[patient_id][3]['static'] = temp_data[patient_id]['sex']
        new_dataset[patient_id][4]['static'] = temp_data[patient_id]['age']
    Functions.pickle_save_object('/home/zhoul0a/Desktop/longxi_platform/delat_project/dataset.pickle', new_dataset)
