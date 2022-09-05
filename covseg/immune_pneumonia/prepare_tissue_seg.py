import os
import numpy as np
import prediction.predict_rescaled as tissue_seg
import visualization.fast_link as visualize
import Tool_Functions.Functions as Functions

rescaled_array_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/rescaled_array/'
array_name_list = os.listdir(rescaled_array_dict)
image_save_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/visualize/'
mask_save_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/tissue_masks/'

for array_name in array_name_list:
    print(array_name)
    if os.path.exists(image_save_dict + 'seg_check/' + array_name[:-4] + '.png'):
        print("processed")
        continue
    rescaled_array = np.load(rescaled_array_dict + array_name)
    lung_mask = tissue_seg.predict_lung_masks_rescaled_array(rescaled_array)
    airway_mask = tissue_seg.get_prediction_airway(rescaled_array, lung_mask=lung_mask)
    blood_vessel_mask = tissue_seg.get_prediction_blood_vessel(rescaled_array, lung_mask=lung_mask)

    known_infection = tissue_seg.predict_covid_19_infection_rescaled_array(rescaled_array, lung_mask=lung_mask)

    Functions.save_np_array(mask_save_dict + 'lung/', array_name[:-4], lung_mask, compress=True)
    Functions.save_np_array(mask_save_dict + 'airway/', array_name[:-4], airway_mask, compress=True)
    Functions.save_np_array(mask_save_dict + 'blood_vessel/', array_name[:-4], blood_vessel_mask, compress=True)
    Functions.save_np_array(mask_save_dict + 'known_lesion/', array_name[:-4], blood_vessel_mask, compress=True)

    highlighted = visualize.visualize_lung_tissues(rescaled_array, lung_mask, airway_mask, blood_vessel_mask)
    Functions.image_save(highlighted[:, :, 256, :], image_save_dict + 'seg_check/' + array_name[:-4],
                         high_resolution=True)
