"""
this is for refined prediction of:
lungs, airways, blood vessels, heart
"""

import Tool_Functions.Functions as Functions
import numpy as np
import prediction.predict_rescaled as predictor
import os


def pipeline_prediction_chest_tissues(rescaled_array_top_dict, mask_top_dict):
    rescaled_array_name_list = os.listdir(rescaled_array_top_dict)
    
    num_arrays = len(rescaled_array_name_list)
    processed_count = 0
    
    for array_name in rescaled_array_name_list:
        print("processing", array_name)
        print(num_arrays - processed_count, 'left')
        
        lung_processed = False
        heart_processed = False
        airway_processed = False
        blood_processed = False
        
        if os.path.exists(mask_top_dict + '/lung/' + array_name[:-4] + '.npz'):
            lung_processed = True
        if os.path.exists(mask_top_dict + '/heart/' + array_name[:-4] + '.npz'):
            heart_processed = True
        if os.path.exists(mask_top_dict + '/airway/' + array_name[:-4] + '.npz'):
            airway_processed = True
        if os.path.exists(mask_top_dict + '/blood/' + array_name[:-4] + '.npz'):
            blood_processed = True
        
        if lung_processed and heart_processed and airway_processed and blood_processed:
            print("all processed")
            processed_count += 1
            continue
        
        rescaled_array = np.load(rescaled_array_top_dict + array_name)
        
        if lung_processed:
            lung = np.load(mask_top_dict + '/lung/' + array_name[:-4] + '.npz')['array']
        else:
            lung = predictor.predict_lung_masks_rescaled_array(rescaled_array)
            Functions.save_np_array(mask_top_dict + '/lung/', array_name[:-4], lung, True)
            
        if not heart_processed:
            heart = predictor.predict_heart_rescaled_array(rescaled_array)
            Functions.save_np_array(mask_top_dict + '/heart/', array_name[:-4], heart, True)
        
        if not airway_processed:
            airway = predictor.get_prediction_airway(rescaled_array, lung_mask=lung)
            Functions.save_np_array(mask_top_dict + '/airway/', array_name[:-4], airway, True)
        
        if not blood_processed:
            blood = predictor.get_prediction_blood_vessel(rescaled_array, lung_mask=lung)
            Functions.save_np_array(mask_top_dict + '/blood/', array_name[:-4], blood, True)
        processed_count += 1
        
        
if __name__ == '__main__':
    top_dict_rescaled_arrays = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/9肺癌/'
    top_dict_mask = '/home/zhoul0a/Desktop/NMI_revision_data/masks/9肺癌/'
    pipeline_prediction_chest_tissues(top_dict_rescaled_arrays, top_dict_mask)

    top_dict_rescaled_arrays = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/pulmonary_nodules/'
    top_dict_mask = '/home/zhoul0a/Desktop/NMI_revision_data/masks/pulmonary_nodules/'
    pipeline_prediction_chest_tissues(top_dict_rescaled_arrays, top_dict_mask)

    top_dict_rescaled_arrays = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/immune_pneumonia/'
    top_dict_mask = '/home/zhoul0a/Desktop/NMI_revision_data/masks/immune_pneumonia/'
    pipeline_prediction_chest_tissues(top_dict_rescaled_arrays, top_dict_mask)

    top_dict_rescaled_arrays = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/new_follow_up/'
    top_dict_mask = '/home/zhoul0a/Desktop/NMI_revision_data/masks/new_follow_up/'
    pipeline_prediction_chest_tissues(top_dict_rescaled_arrays, top_dict_mask)
