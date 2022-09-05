import numpy as np

import os

import Tool_Functions.Functions as Functions

import format_convert.read_in_CT as read_in
import format_convert.spatial_normalize as spatial_normalize


enhanced = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/progonosis_enhanced/xghf-25_2020-07-29.npz')['array']
mask = np.load('/home/zhoul0a/Desktop/prognosis_project/visible&invisible_lesions/invisible_lesion/xghf-25_2020-07-29.npz')['array']
enhance_slice = np.clip(enhanced[:, :, 308], 0, 0.1)
Functions.image_show(np.clip(enhanced[:, :, 308], 0, 0.1), gray=True)
Functions.merge_image_with_mask(enhance_slice, mask[:, :, 308], save_path='/home/zhoul0a/Desktop/transfer/example.png', high_resolution=True)

exit()
rescaled_ct = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/xghf-25_2020-07-29.npy')
lung = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/lung_masks/xghf-25/xghf-25_2020-07-29_mask_refine.npz')['array']
airway = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/air_way_mask_stage_two/xghf-25/xghf-25_2020-07-29_mask_refine.npz')['array']
blood_vessel = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/blood_vessel_mask_stage_two/xghf-25/xghf-25_2020-07-29_mask_refine.npz')['array']

import post_processing.parenchyma_enhancement as enhance
enhanced = enhance.enhance_with_quantile(rescaled_ct, lung, airway, blood_vessel)
Functions.image_show(enhanced[:, :, 308], gray=True)

exit()

