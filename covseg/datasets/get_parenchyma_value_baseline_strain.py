import os

import numpy as np

import post_processing.parenchyma_enhancement as enhance

import Tool_Functions.Functions as Functions

array_name_set = set(os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/rescaled_arrays'))

mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/masks/'
rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/previous_strain/rescaled_arrays/'

total_number = len(array_name_set)
processed = 0

if os.path.exists('/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_previous.pickle'):
    parenchyma_dict = Functions.pickle_load_object(
        '/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_previous.pickle')
    print('continue with', len(list(parenchyma_dict.keys())))
    print(parenchyma_dict)
    total_number = total_number - len(list(parenchyma_dict.keys()))
else:
    parenchyma_dict = {}

processed_list = list(parenchyma_dict.keys())

for array_name in array_name_set:
    print('processing:', array_name[:-4], total_number - processed, 'left\n')

    if array_name[:-4] in processed_list:
        print("processed")
        processed += 1
        continue

    array = np.load(rescaled_array_top_dict + array_name[:-4] + '.npy')

    lungs = np.load(mask_top_dict + 'lung_mask/' + array_name[:-4] + '.npz')['array']

    airways = np.load(mask_top_dict + 'airway/' + array_name[:-4] + '.npz')['array']
    blood = np.load(mask_top_dict + 'blood_vessel/' + array_name[:-4] + '.npz')['array']

    enhanced_array, parenchyma_baseline = enhance.remove_airway_and_blood_vessel_general_sampling(
        array, lungs, airways, blood, parenchyma_value=True)

    parenchyma_dict[array_name[:-4]] = parenchyma_baseline

    Functions.pickle_save_object('/home/zhoul0a/Desktop/COVID-19 delta/dataset/parenchyma_previous.pickle',
                                 parenchyma_dict)
    processed += 1
