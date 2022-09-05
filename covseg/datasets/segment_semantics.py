import os

import prediction.predict_rescaled as predictor

import numpy as np

import post_processing.parenchyma_enhancement as enhance

import Tool_Functions.Functions as Functions

array_list = os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays')
mask_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/masks/'
rescaled_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays/'
enhance_array_top_dict = '/home/zhoul0a/Desktop/COVID-19 delta/enhanced_array/'
total_number = len(array_list)
processed = 0

for array_name in array_list:
    print('processing:', array_name, total_number - processed, 'left\n')

    s_lung = os.path.exists(mask_top_dict + 'lung_mask/' + array_name[:-4] + '.npz')
    s_lesion = os.path.exists(mask_top_dict + 'lesions/' + array_name[:-4] + '.npz')
    s_airway = os.path.exists(mask_top_dict + 'airway/' + array_name[:-4] + '.npz')
    s_blood = os.path.exists(mask_top_dict + 'blood_vessel/' + array_name[:-4] + '.npz')
    s_enhance = os.path.exists(enhance_array_top_dict + array_name[:-4] + '.npz')

    print('lung', s_lung, 'lesion', s_lesion, 'airway', s_airway, 'blood', s_blood, 'enhance', s_enhance)
    if s_lung and s_lesion and s_airway and s_blood and s_enhance:
        print("processed")
        processed += 1
        continue

    array = np.load(rescaled_array_top_dict + array_name)

    if s_lung is True:
        lungs = np.load(mask_top_dict + 'lung_mask/' + array_name[:-4] + '.npz')['array']
    else:
        lungs = predictor.predict_lung_masks_rescaled_array(array)

        Functions.save_np_array(mask_top_dict + 'lung_mask/', array_name[:-4], lungs, True)

    lesions = predictor.predict_covid_19_infection_rescaled_array(array, lung_mask=lungs, threshold=0.8)

    Functions.save_np_array(mask_top_dict + 'lesions/', array_name[:-4], lesions, True)

    if s_airway is False:
        airways = predictor.get_prediction_airway(array, lung_mask=lungs)

        Functions.save_np_array(mask_top_dict + 'airway/', array_name[:-4], airways, True)

    if s_blood is False:
        blood = predictor.get_prediction_blood_vessel(array, lung_mask=lungs)

        Functions.save_np_array(mask_top_dict + 'blood_vessel/', array_name[:-4], blood, True)

    if s_enhance is False:
        airways = np.load(mask_top_dict + 'airway/' + array_name[:-4] + '.npz')['array']
        blood = np.load(mask_top_dict + 'blood_vessel/' + array_name[:-4] + '.npz')['array']

        enhanced_array = enhance.remove_airway_and_blood_vessel_general_sampling(array, lungs, airways, blood)

        Functions.save_np_array(enhance_array_top_dict, array_name[:-4], enhanced_array,
                                True)
    processed += 1
