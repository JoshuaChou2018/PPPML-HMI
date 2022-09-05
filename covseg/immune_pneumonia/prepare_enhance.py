import os
import numpy as np
import post_processing.parenchyma_enhancement as enhance
import Tool_Functions.Functions as Functions

rescaled_array_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/rescaled_array/'
array_name_list = os.listdir(rescaled_array_dict)
image_save_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/visualize/'
mask_save_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/tissue_masks/'

for array_name in array_name_list:
    print(array_name)
    if os.path.exists(image_save_dict + 'enhance_check/' + array_name[:-4] + '.png'):
        print("processed")
        continue
    rescaled_array = np.load(rescaled_array_dict + array_name)
    lung_mask = np.load(mask_save_dict + 'lung/' + array_name[:-4] + '.npz')['array']
    airway_mask = np.load(mask_save_dict + 'airway/' + array_name[:-4] + '.npz')['array']
    blood_vessel_mask = np.load(mask_save_dict + 'blood_vessel/' + array_name[:-4] + '.npz')['array']
    known_lesion = np.load(mask_save_dict + 'known_lesion/' + array_name[:-4] + '.npz')['array']

    enhanced_array = enhance.enhance_with_quantile(rescaled_array, lung_mask, airway_mask, blood_vessel_mask,
                                                   known_lesion)
    Functions.image_save(enhanced_array[:, :, 256], image_save_dict + 'enhance_check/' + array_name[:-4],
                         high_resolution=True, gray=True)
