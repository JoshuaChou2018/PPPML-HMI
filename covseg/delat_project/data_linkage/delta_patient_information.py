"""
manage the dataset, which is freq dict with key: patient_id
dataset[patient_id] is also freq dict, with key: age, sex, vaccine, clinical_severity, strain, discharge, radiomics
age: an int
sex: 'M' for male and 'F' for female
vaccine: an int, number of vaccine
clinical severity: an int, three level 0, 1, 2, 3 for mild, middle, severe, critical
    Highest seven-category scale during hospital stay: scale 1-2 (level 0), scale 3 (level 1), scale 4 (level 2),
    scale 5-6 (level 3)
strain: 'delta', 'baseline'
discharge: True, False, which indicate whether the last CT is qualified for discharge
radiomics: freq dict, with key: scan time like '20210530';
dataset[patient_id][radiomics][time] is freq dict, with key:
lesion_ratio, lesion_severity_ct, lesion_severity_enhance,
lesion_component_count, lesion_CT_signal_probability_distribution_list;
CT_quality: good, normal, poor
"""

import numpy as np
import Tool_Functions.Functions as Functions
import os

dataset_path = '/home/zhoul0a/Desktop/COVID-19 delta/dataset/dataset.pickle'
if os.path.exists(dataset_path):
    dataset = Functions.pickle_load_object(dataset_path)
else:
    dataset = {}

key_set = set(dataset.keys())


def complete_personal_information(patient_id, clinical_severity, sex, age, vaccine, discharge=True, strain='delta'):
    global dataset, key_set
    assert 0 <= clinical_severity <= 4
    assert type(discharge) is bool
    assert sex in ['M', 'F']
    assert 1 <= age < 110
    assert 0 <= vaccine <= 5
    assert type(strain) is str
    if patient_id not in key_set:
        dataset[patient_id] = {}
        key_set.add(patient_id)
    dataset[patient_id]['clinical_severity'] = clinical_severity
    dataset[patient_id]['discharge'] = discharge
    dataset[patient_id]['sex'] = sex
    dataset[patient_id]['discharge'] = discharge
    dataset[patient_id]['age'] = age
    dataset[patient_id]['vaccine'] = vaccine
    dataset[patient_id]['strain'] = strain


def assign_for_delta_dataset():
    complete_personal_information('P0140328', clinical_severity=0, discharge=True, sex='M', age=9, vaccine=0)

    complete_personal_information('P0140329', clinical_severity=0, discharge=True, sex='F', age=40, vaccine=2)

    complete_personal_information('P0140382', clinical_severity=0, discharge=True, sex='M', age=7, vaccine=0)

    complete_personal_information('P0140383', clinical_severity=0, discharge=False, sex='M', age=31, vaccine=2)

    complete_personal_information('P0140455', clinical_severity=0, discharge=False, sex='M', age=20, vaccine=2)

    complete_personal_information('P0140516', clinical_severity=0, discharge=False, sex='F', age=15, vaccine=0)

    complete_personal_information('P0140538', clinical_severity=0, discharge=True, sex='M', age=18, vaccine=0)

    complete_personal_information('P0140541', clinical_severity=0, discharge=True, sex='F', age=53, vaccine=2)

    complete_personal_information('P0140683', clinical_severity=0, discharge=True, sex='F', age=7, vaccine=0)

    complete_personal_information('P0140684', clinical_severity=0, discharge=True, sex='M', age=36, vaccine=2)

    complete_personal_information('P0140711', clinical_severity=0, discharge=False, sex='M', age=36, vaccine=0)

    complete_personal_information('P0140737', clinical_severity=0, discharge=False, sex='M', age=37, vaccine=2)

    complete_personal_information('P0140768', clinical_severity=0, discharge=False, sex='F', age=48, vaccine=1)

    complete_personal_information('P0140790', clinical_severity=0, discharge=False, sex='M', age=17, vaccine=0)  #

    complete_personal_information('P0140794', clinical_severity=0, discharge=False, sex='F', age=22, vaccine=1)

    complete_personal_information('P0140904', clinical_severity=0, discharge=True, sex='M', age=13, vaccine=0)

    complete_personal_information('P0140905', clinical_severity=0, discharge=True, sex='F', age=34, vaccine=2)

    complete_personal_information('P0140906', clinical_severity=0, discharge=True, sex='F', age=57, vaccine=2)

    complete_personal_information('P0140384', clinical_severity=0, discharge=False, sex='M', age=40, vaccine=2)

    complete_personal_information('P0141005', clinical_severity=0, discharge=False, sex='F', age=8, vaccine=0)

    complete_personal_information('P0140688', clinical_severity=0, discharge=False, sex='F', age=42, vaccine=2)  #

    complete_personal_information('P0054885', clinical_severity=1, discharge=False, sex='M', age=35, vaccine=1)

    complete_personal_information('P0106332', clinical_severity=1, discharge=False, sex='F', age=56, vaccine=2)

    complete_personal_information('P0107156', clinical_severity=1, discharge=False, sex='F', age=51, vaccine=1)

    complete_personal_information('P0140453', clinical_severity=1, discharge=False, sex='F', age=7, vaccine=0)

    complete_personal_information('P0140454', clinical_severity=1, discharge=False, sex='M', age=30, vaccine=1)

    complete_personal_information('P0140463', clinical_severity=1, discharge=False, sex='F', age=34, vaccine=2)

    complete_personal_information('P0140488', clinical_severity=1, discharge=True, sex='M', age=49, vaccine=1)

    complete_personal_information('P0140489', clinical_severity=1, discharge=False, sex='M', age=9, vaccine=0)

    complete_personal_information('P0140509', clinical_severity=1, discharge=False, sex='F', age=19, vaccine=1)

    complete_personal_information('P0140573', clinical_severity=1, discharge=False, sex='F', age=58, vaccine=1)

    complete_personal_information('P0140575', clinical_severity=1, discharge=False, sex='M', age=15, vaccine=0)

    complete_personal_information('P0140588', clinical_severity=1, discharge=False, sex='F', age=28, vaccine=1)

    complete_personal_information('P0140589', clinical_severity=1, discharge=False, sex='M', age=9, vaccine=0)

    complete_personal_information('P0140590', clinical_severity=1, discharge=False, sex='F', age=19, vaccine=1)

    complete_personal_information('P0140677', clinical_severity=1, discharge=False, sex='F', age=39, vaccine=1)

    complete_personal_information('P0140678', clinical_severity=1, discharge=False, sex='F', age=51, vaccine=2)

    complete_personal_information('P0140682', clinical_severity=1, discharge=False, sex='M', age=65, vaccine=0)

    complete_personal_information('P0140686', clinical_severity=1, discharge=False, sex='F', age=5, vaccine=0)

    complete_personal_information('P0140706', clinical_severity=1, discharge=False, sex='M', age=35, vaccine=2)

    complete_personal_information('P0140718', clinical_severity=1, discharge=False, sex='M', age=48, vaccine=2)

    complete_personal_information('P0140719', clinical_severity=1, discharge=False, sex='F', age=14, vaccine=0)

    complete_personal_information('P0140748', clinical_severity=1, discharge=False, sex='F', age=12, vaccine=0)

    complete_personal_information('P0140791', clinical_severity=1, discharge=False, sex='M', age=54, vaccine=0)

    complete_personal_information('P0140848', clinical_severity=1, discharge=False, sex='F', age=63, vaccine=0)

    complete_personal_information('P0106332', clinical_severity=1, discharge=False, sex='F', age=56, vaccine=2)

    complete_personal_information('P0107156', clinical_severity=1, discharge=False, sex='F', age=51, vaccine=1)

    complete_personal_information('P0140517', clinical_severity=1, discharge=False, sex='F', age=17, vaccine=0)

    complete_personal_information('P0140542', clinical_severity=1, discharge=False, sex='M', age=8, vaccine=0)

    complete_personal_information('P0140596', clinical_severity=1, discharge=False, sex='F', age=12, vaccine=0)

    complete_personal_information('P0140903', clinical_severity=1, discharge=False, sex='M', age=12, vaccine=0)

    complete_personal_information('P0141419', clinical_severity=1, discharge=False, sex='F', age=26, vaccine=1)

    complete_personal_information('P0140767', clinical_severity=1, discharge=False, sex='M', age=19, vaccine=2)

    complete_personal_information('P0140457', clinical_severity=2, discharge=False, sex='F', age=59, vaccine=2)

    complete_personal_information('P0140514', clinical_severity=2, discharge=False, sex='F', age=61, vaccine=0)

    complete_personal_information('P0140555', clinical_severity=2, discharge=False, sex='F', age=71, vaccine=0)

    complete_personal_information('P0140614', clinical_severity=2, discharge=False, sex='M', age=45, vaccine=2)

    complete_personal_information('P0140633', clinical_severity=2, discharge=False, sex='M', age=38, vaccine=2)

    complete_personal_information('P0140679', clinical_severity=2, discharge=False, sex='F', age=57, vaccine=2)

    complete_personal_information('P0140685', clinical_severity=2, discharge=False, sex='F', age=30, vaccine=2)

    complete_personal_information('P0140744', clinical_severity=2, discharge=False, sex='F', age=52, vaccine=2)

    complete_personal_information('P0140811', clinical_severity=2, discharge=False, sex='F', age=58, vaccine=0)

    complete_personal_information('P0140828', clinical_severity=2, discharge=False, sex='M', age=60, vaccine=1)

    complete_personal_information('P0140866', clinical_severity=2, discharge=False, sex='M', age=41, vaccine=0)

    complete_personal_information('P0140881', clinical_severity=2, discharge=False, sex='F', age=59, vaccine=2)

    complete_personal_information('P0140555', clinical_severity=2, discharge=False, sex='F', age=71, vaccine=0)

    complete_personal_information('P0140173', clinical_severity=2, discharge=True, sex='F', age=38, vaccine=2)

    complete_personal_information('P0140595', clinical_severity=2, discharge=False, sex='M', age=39, vaccine=1)

    complete_personal_information('P0058339', clinical_severity=2, discharge=False, sex='F', age=32, vaccine=0)

    complete_personal_information('P0140521', clinical_severity=2, discharge=False, sex='F', age=38, vaccine=0)  #

    complete_personal_information('P0140557', clinical_severity=2, discharge=False, sex='M', age=22, vaccine=1)  #

    complete_personal_information('P0140187', clinical_severity=0, discharge=True, sex='F', age=37, vaccine=1)

    complete_personal_information('P0140264', clinical_severity=1, discharge=True, sex='M', age=47, vaccine=2)

    complete_personal_information('P0140268', clinical_severity=2, discharge=True, sex='F', age=44, vaccine=2)

    complete_personal_information('P0140270', clinical_severity=1, discharge=True, sex='M', age=33, vaccine=2)

    complete_personal_information('P0140271', clinical_severity=0, discharge=True, sex='F', age=34, vaccine=2)

    complete_personal_information('P0140360', clinical_severity=1, discharge=True, sex='F', age=43, vaccine=1)


def assign_for_baseline_dataset():
    complete_personal_information('baiyanqiu', clinical_severity=0, strain='baseline', sex='F', age=42, vaccine=0)

    complete_personal_information('chen jing', clinical_severity=0, strain='baseline', sex='F', age=30, vaccine=0)

    complete_personal_information('cui jin ling', clinical_severity=0, strain='baseline', sex='F', age=49, vaccine=0)

    complete_personal_information('cui ying', clinical_severity=0, strain='baseline', sex='F', age=38, vaccine=0)

    complete_personal_information('gao zhen bing', clinical_severity=0, strain='baseline', sex='M', age=54, vaccine=0)

    complete_personal_information('HU CHUN GUO', clinical_severity=0, strain='baseline', sex='M', age=41, vaccine=0)

    complete_personal_information('huangyuliang', clinical_severity=0, strain='baseline', sex='M', age=33, vaccine=0)

    complete_personal_information('jiang ya bo', clinical_severity=0, strain='baseline', sex='M', age=47, vaccine=0)

    complete_personal_information('liu jia hui', clinical_severity=0, strain='baseline', sex='F', age=25, vaccine=0)

    complete_personal_information('LIU JIA LI', clinical_severity=0, strain='baseline', sex='F', age=48, vaccine=0)

    complete_personal_information('qi qi', clinical_severity=0, strain='baseline', sex='F', age=38, vaccine=0)

    complete_personal_information('SUN WEN CHANG', clinical_severity=0, strain='baseline', sex='M', age=50, vaccine=0)

    complete_personal_information('zhang lin', clinical_severity=0, strain='baseline', sex='M', age=34, vaccine=0)

    complete_personal_information('zhang yun long', clinical_severity=0, strain='baseline', sex='M', age=19, vaccine=0)

    complete_personal_information('chen mei fang', clinical_severity=0, strain='baseline', sex='F', age=36, vaccine=0)

    complete_personal_information('CHEN ZE KAI', clinical_severity=0, strain='baseline', sex='M', age=23, vaccine=0)

    complete_personal_information('chen zi hang', clinical_severity=0, strain='baseline', sex='M', age=20, vaccine=0)

    complete_personal_information('DAI JIAN GUO', clinical_severity=0, strain='baseline', sex='M', age=49, vaccine=0)

    complete_personal_information('fu guo qiang', clinical_severity=0, strain='baseline', sex='M', age=44, vaccine=0)

    complete_personal_information('GUO YONG', clinical_severity=0, strain='baseline', sex='M', age=34, vaccine=0)

    complete_personal_information('shi chun lei', clinical_severity=0, strain='baseline', sex='M', age=40, vaccine=0)

    complete_personal_information('chen lian ming', clinical_severity=1, strain='baseline', sex='F', age=60, vaccine=0)

    complete_personal_information('CHEN ZHI HAO', clinical_severity=1, strain='baseline', sex='M', age=31, vaccine=0)

    complete_personal_information('fu zhong chen', clinical_severity=1, strain='baseline', sex='M', age=72, vaccine=0)

    complete_personal_information('gao ji bao', clinical_severity=1, strain='baseline', sex='M', age=46, vaccine=0)

    complete_personal_information('GONG QING FU', clinical_severity=1, strain='baseline', sex='M', age=63, vaccine=0)

    complete_personal_information('gong wen jing', clinical_severity=1, strain='baseline', sex='F', age=25, vaccine=0)

    complete_personal_information('guo shi bo', clinical_severity=1, strain='baseline', sex='F', age=52, vaccine=0)

    complete_personal_information('li cheng mei', clinical_severity=1, strain='baseline', sex='F', age=46, vaccine=0)

    complete_personal_information('LIU LEI MEI', clinical_severity=1, strain='baseline', sex='F', age=47, vaccine=0)

    complete_personal_information('chen jian song', clinical_severity=1, strain='baseline', sex='M', age=59, vaccine=0)

    complete_personal_information('DOU YING JIE', clinical_severity=2, strain='baseline', sex='F', age=63, vaccine=0)

    complete_personal_information('gao xi yun', clinical_severity=2, strain='baseline', sex='F', age=58, vaccine=0)

    complete_personal_information('jing su bo', clinical_severity=2, strain='baseline', sex='F', age=51, vaccine=0)

    complete_personal_information('gao zhong tao', clinical_severity=2, strain='baseline', sex='M', age=36, vaccine=0)

    complete_personal_information('sui hong yan', clinical_severity=2, strain='baseline', sex='F', age=47, vaccine=0)

    complete_personal_information('zhang guo chun', clinical_severity=2, strain='baseline', sex='M', age=48, vaccine=0)

    complete_personal_information('ma xue yong', clinical_severity=2, strain='baseline', sex='M', age=58, vaccine=0)

    complete_personal_information('pan yan guo', clinical_severity=2, strain='baseline', sex='M', age=34, vaccine=0)

    complete_personal_information('zhang rong cai', clinical_severity=2, strain='baseline', sex='M', age=41, vaccine=0)

    complete_personal_information('zhang xiu xiang', clinical_severity=2, strain='baseline', sex='M', age=66, vaccine=0)

    complete_personal_information('bai-shao-fu', clinical_severity=1, strain='baseline', sex='M', age=63, vaccine=0)

    complete_personal_information('bo-huai-ying', clinical_severity=1, strain='baseline', sex='F', age=67, vaccine=0)

    complete_personal_information('cang-ya-nan', clinical_severity=1, strain='baseline', sex='M', age=71, vaccine=0)

    complete_personal_information('tang-hai-feng', clinical_severity=1, strain='baseline', sex='M', age=77, vaccine=0)

    complete_personal_information('wang-jian-xin', clinical_severity=1, strain='baseline', sex='M', age=24, vaccine=0)

    complete_personal_information('feng-xiao', clinical_severity=2, strain='baseline', sex='M', age=59, vaccine=0)

    complete_personal_information('hou-fang-cai', clinical_severity=2, strain='baseline', sex='M', age=57, vaccine=0)

    complete_personal_information('hu-zhi-xin', clinical_severity=2, strain='baseline', sex='M', age=40, vaccine=0)

    complete_personal_information('lv-xi-you', clinical_severity=2, strain='baseline', sex='M', age=48, vaccine=0)

    complete_personal_information('liu-gui-lan', clinical_severity=2, strain='baseline', sex='F', age=65, vaccine=0)

    complete_personal_information('wang-li-ming', clinical_severity=1, strain='baseline', sex='M', age=56, vaccine=0)

    complete_personal_information('dong-wei', clinical_severity=2, strain='baseline', sex='M', age=52, vaccine=0)

    complete_personal_information('li-zhi-feng', clinical_severity=2, strain='baseline', sex='M', age=56, vaccine=0)

    complete_personal_information('zhang-xiao-juan', clinical_severity=2, strain='baseline', sex='F', age=68, vaccine=0)

    complete_personal_information('xgfy-002', clinical_severity=1, strain='baseline', sex='F', age=60, vaccine=0)

    complete_personal_information('xgfy-003', clinical_severity=2, strain='baseline', sex='F', age=56, vaccine=0)

    complete_personal_information('xgfy-006', clinical_severity=0, strain='baseline', sex='F', age=37, vaccine=0)

    complete_personal_information('xgfy-007', clinical_severity=1, strain='baseline', sex='F', age=56, vaccine=0)

    complete_personal_information('xgfy-016', clinical_severity=2, strain='baseline', sex='F', age=57, vaccine=0)

    complete_personal_information('xgfy-022', clinical_severity=0, strain='baseline', sex='M', age=48, vaccine=0)

    complete_personal_information('xgfy-029', clinical_severity=0, strain='baseline', sex='F', age=69, vaccine=0)

    complete_personal_information('xgfy-033', clinical_severity=0, strain='baseline', sex='M', age=61, vaccine=0)

    complete_personal_information('xgfy-036', clinical_severity=1, strain='baseline', sex='M', age=57, vaccine=0)

    complete_personal_information('xgfy-037', clinical_severity=2, strain='baseline', sex='F', age=47, vaccine=0)

    complete_personal_information('xgfy-041', clinical_severity=1, strain='baseline', sex='F', age=65, vaccine=0)

    complete_personal_information('xgfy-044', clinical_severity=0, strain='baseline', sex='F', age=66, vaccine=0)

    complete_personal_information('xgfy-046', clinical_severity=0, strain='baseline', sex='F', age=54, vaccine=0)

    complete_personal_information('xgfy-047', clinical_severity=0, strain='baseline', sex='F', age=30, vaccine=0)

    complete_personal_information('xgfy-048', clinical_severity=1, strain='baseline', sex='M', age=44, vaccine=0)

    complete_personal_information('xgfy-051', clinical_severity=0, strain='baseline', sex='M', age=58, vaccine=0)

    complete_personal_information('xgfy-058', clinical_severity=0, strain='baseline', sex='F', age=46, vaccine=0)

    complete_personal_information('xgfy-059', clinical_severity=1, strain='baseline', sex='M', age=69, vaccine=0)

    complete_personal_information('xgfy-062', clinical_severity=1, strain='baseline', sex='M', age=67, vaccine=0)

    complete_personal_information('xgfy-064', clinical_severity=0, strain='baseline', sex='M', age=62, vaccine=0)

    complete_personal_information('liu-tie-ming', clinical_severity=1, strain='baseline', sex='M', age=51, vaccine=0)

    complete_personal_information('liang-shu-jun', clinical_severity=1, strain='baseline', sex='M', age=64, vaccine=0)


def add_lesion_ratio_and_severity():
    parenchyma_dict_delta = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_delta.pickle')
    parenchyma_dict_previous = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_previous.pickle')

    delta_scan_name_set = set(parenchyma_dict_delta.keys())
    previous_scan_name_set = set(parenchyma_dict_previous.keys())

    suitable_for_radiomics = delta_scan_name_set | previous_scan_name_set

    patient_with_info = set(dataset.keys())

    if 'radiomics_processed_list' not in list(dataset.keys()):
        dataset['radiomics_processed_list'] = []
    print(dataset['radiomics_processed_list'])

    total_scans = len(suitable_for_radiomics)
    processed = 0
    for id_time in suitable_for_radiomics:
        patient_id, time = id_time.split('_')
        if patient_id in patient_with_info:  # we calculate the radiomics
            print('processing', patient_id, time, 'number_left', total_scans - processed)
            if id_time in dataset['radiomics_processed_list']:
                print('processed')
                processed += 1
                continue

            if id_time in delta_scan_name_set:
                mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/masks/'
                rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/'
                baseline = parenchyma_dict_delta[id_time]
            else:
                mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/masks/'
                rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/rescaled_arrays/'
                baseline = parenchyma_dict_previous[id_time]

            lung_mask = np.load(mask_top_dict + 'lung_mask/' + id_time + '.npz')['array']
            lesion_mask = np.load(mask_top_dict + 'lesions/' + id_time + '.npz')['array']
            blood_vessel = np.load(mask_top_dict + 'blood_vessel/' + id_time + '.npz')['array']

            rescaled_array = np.load(rescaled_array_top_dict + id_time + '.npy')

            lung_volume = np.sum(lung_mask)
            lesion_volume = np.sum(lesion_mask)

            lesion_ratio = lesion_volume / lung_volume

            lesion_region = rescaled_array * (1 - blood_vessel) * lesion_mask
            total_lesion_ct = np.sum(lesion_region)
            lesion_severity_ct = total_lesion_ct / lung_volume
            lesion_severity_enhance = (total_lesion_ct - lesion_volume * baseline) / lung_volume

            print('lesion_ratio', lesion_ratio, 'lesion_severity_ct', lesion_severity_ct,
                  'lesion_severity_enhance', lesion_severity_enhance)

            if 'lesion_ratio' not in list(dataset[patient_id].keys()):
                dataset[patient_id]['lesion_ratio'] = {}
            if 'lesion_severity_ct' not in list(dataset[patient_id].keys()):
                dataset[patient_id]['lesion_severity_ct'] = {}
            if 'lesion_severity_enhance' not in list(dataset[patient_id].keys()):
                dataset[patient_id]['lesion_severity_enhance'] = {}

            dataset[patient_id]['lesion_ratio'][time] = lesion_ratio
            dataset[patient_id]['lesion_severity_ct'][time] = lesion_severity_ct
            dataset[patient_id]['lesion_severity_enhance'][time] = lesion_severity_enhance

            dataset['radiomics_processed_list'].append(id_time)

            Functions.pickle_save_object(dataset_path, dataset)

            processed += 1
        else:
            print('this scan do not have patient info')
            processed += 1


def add_spatial_distribution():
    parenchyma_dict_delta = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_delta.pickle')
    parenchyma_dict_previous = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_previous.pickle')

    delta_scan_name_set = set(parenchyma_dict_delta.keys())
    previous_scan_name_set = set(parenchyma_dict_previous.keys())

    suitable_for_radiomics = delta_scan_name_set | previous_scan_name_set

    patient_with_info = set(dataset.keys())

    if 'spatial_processed_list' not in list(dataset.keys()):
        dataset['spatial_processed_list'] = []
    print(dataset['spatial_processed_list'])

    total_scans = len(suitable_for_radiomics)
    processed = 0

    suitable_for_radiomics.remove('tang-hai-feng_20210219')
    suitable_for_radiomics.remove('tang-hai-feng_20210227')

    for id_time in suitable_for_radiomics:
        patient_id, time = id_time.split('_')
        if patient_id in patient_with_info:  # we calculate the radiomics
            print('processing', patient_id, time, 'number_left', total_scans - processed)
            if id_time in dataset['spatial_processed_list']:
                print('processed')
                processed += 1
                continue

            if id_time in delta_scan_name_set:
                mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/masks/'
                rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/'
                branching_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/branching_array/'
            else:
                mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/masks/'
                rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/rescaled_arrays/'
                branching_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/branching_array/'

            branching = np.load(branching_array_top_dict + id_time + '.npz')['array']
            upper_region = np.array(branching >= 0.1, 'float32') * np.array(branching < 3.5, 'float32')
            mid_region = np.array(branching >= 3.5, 'float32') * np.array(branching < 7, 'float32')
            lower_region = np.array(branching >= 7, 'float32')

            lesion_mask = np.load(mask_top_dict + 'lesions/' + id_time + '.npz')['array']

            rescaled_array = np.load(rescaled_array_top_dict + id_time + '.npy')

            upper_volume = np.sum(upper_region)
            mid_volume = np.sum(mid_region)
            lower_volume = np.sum(lower_region)

            lesion_upper = lesion_mask * upper_region
            lesion_mid = lesion_mask * mid_region
            lesion_lower = lesion_mask * lower_region

            lesion_volume_upper = np.sum(lesion_upper)
            lesion_volume_mid = np.sum(lesion_mid)
            lesion_volume_lower = np.sum(lesion_lower)

            lesion_ratio_upper = lesion_volume_upper / upper_volume
            lesion_ratio_mid = lesion_volume_mid / mid_volume
            lesion_ratio_lower = lesion_volume_lower / lower_volume

            lesion_severity_upper = np.sum(rescaled_array * lesion_upper) / upper_volume
            lesion_severity_mid = np.sum(rescaled_array * lesion_mid) / mid_volume
            lesion_severity_lower = np.sum(rescaled_array * lesion_lower) / lower_volume

            print('lesion_ratios', lesion_ratio_upper, lesion_ratio_mid, lesion_ratio_lower,
                  'lesion_severities', lesion_severity_upper, lesion_severity_mid, lesion_severity_lower)

            if 'lesion_ratio_distribution' not in list(dataset[patient_id].keys()):
                dataset[patient_id]['lesion_ratio_distribution'] = {}
            if 'lesion_severity_distribution' not in list(dataset[patient_id].keys()):
                dataset[patient_id]['lesion_severity_distribution'] = {}

            dataset[patient_id]['lesion_ratio_distribution'][time] = \
                [lesion_ratio_upper, lesion_ratio_mid, lesion_ratio_lower]
            dataset[patient_id]['lesion_severity_distribution'][time] = \
                [lesion_severity_upper, lesion_severity_mid, lesion_severity_lower]

            dataset['spatial_processed_list'].append(id_time)

            Functions.pickle_save_object(dataset_path, dataset)

            processed += 1
        else:
            print('this scan do not have patient info')
            processed += 1


def date_passed_from_20200101(date_calendar='20200328'):
    if not isinstance(date_calendar, str):
        date_calendar = str(date_calendar)
    assert len(date_calendar) == 8
    if date_calendar[0:4] == '2020':
        date_every_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        date_every_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        assert date_calendar[0:4] == '2021'
    month = int(date_calendar[4:6])
    assert 0 < month < 13
    date = int(date_calendar[6::])
#    assert 0 < date <= date_every_month[month - 1]
    if date_calendar[0:4] == '2020':
        return int(date + np.sum(date_every_month[0: month - 1]) - 1)
    else:
        return int(date + np.sum(date_every_month[0: month - 1]) - 1) + 366


def reverse_function_for_date_passed_from_20200101(passed_date=100):
    if passed_date > 365:
        year_2021 = True
        passed_date -= 366
        date_every_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        year_2021 = False
        date_every_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert 0 <= passed_date < 366

    month = 1
    for i in range(0, 12):
        if passed_date - date_every_month[i] >= 0:
            passed_date -= date_every_month[i]
            month += 1
        else:
            break
    passed_date += 1
    if year_2021:
        return '2021' + str(int(month / 10)) + str(month % 10) + str(int(passed_date / 10)) + str(passed_date % 10)
    return '2020' + str(int(month / 10)) + str(month % 10) + str(int(passed_date / 10)) + str(passed_date % 10)


def calendar_time_difference(start_calendar='20200201', end_calendar='20200421'):

    return date_passed_from_20200101(end_calendar) - date_passed_from_20200101(start_calendar)


if __name__ == '__main__':
    patient_list = set(dataset.keys())
    patient_list.remove('radiomics_processed_list')
    patient_list.remove('spatial_processed_list')
    count = 0
    for patient in patient_list:
        print(patient)
        if dataset[patient]['strain'] == 'delta':
            if 18 <= dataset[patient]['age'] <= 60:
                if dataset[patient]['vaccine'] == 2 and dataset[patient]['clinical_severity'] == 2:
                    count += 1
    print(count)
    exit()
    add_spatial_distribution()
    exit()
    print(dataset['P0140633']['lesion_ratio'])
    print(dataset['P0140633']['lesion_severity_ct'])
    print(dataset['P0140633']['lesion_severity_enhance'])
    exit()
    add_lesion_ratio_and_severity()
    exit()
    assign_for_delta_dataset()

    assign_for_baseline_dataset()

    print(len(list(dataset.keys())))
    Functions.pickle_save_object(dataset_path, dataset)
    exit()
    key_list = list(dataset.keys())

    v_0 = 0
    v_1 = 0
    v_2 = 0
    v = 0
    t = 0
    age_list = []

    for key in key_list:
        if key in ['P0140558', 'P0087135']:
            continue
        patient_info = dataset[key]
        print(key)

        if not patient_info['clinical_severity'] == 4:
            t += 1

        if patient_info['age'] >= 18:
            v += 1

        if patient_info['clinical_severity'] == 2:
            if patient_info['age'] > 18:
                if patient_info['vaccine'] == 0:
                    v_0 += 1
                if patient_info['vaccine'] == 1:
                    v_1 += 1
                if patient_info['vaccine'] == 2:
                    v_2 += 1

    print(v_0, v_1, v_2, v, t)
    exit()
    Functions.pickle_save_object(dataset_path, dataset)
