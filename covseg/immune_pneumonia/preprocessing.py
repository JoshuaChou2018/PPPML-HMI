import numpy as np

import os

import Tool_Functions.Functions as Functions

import format_convert.read_in_CT as read_in
import format_convert.spatial_normalize as spatial_normalize


top_dict_raw_data = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/raw_data/'
gt_visualize_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/visualize/gt_check/'
data_visualize_dict = '/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/visualize/data_check/'
patient_id_list = os.listdir(top_dict_raw_data)
for patient_id in patient_id_list:
    print(patient_id)

    patient_data_dict = top_dict_raw_data + patient_id + '/'
    time_list = os.listdir(patient_data_dict)
    for time in time_list:
        print(time)
        if os.path.exists(gt_visualize_dict + patient_id + '_' + time + '.png'):
            if os.path.exists(data_visualize_dict + patient_id + '_' + time + '.png'):
                print('processed')
                continue
        ct_data_dict = patient_data_dict + time + '/Data/raw_data/'
        gt_data_dict = patient_data_dict + time + '/Data/ground_truth/'
        mha_list = os.listdir(gt_data_dict)
        assert len(mha_list) == 1

        ct_data, resolution = read_in.stack_dcm_files(ct_data_dict)
        ground_truth = Functions.read_in_mha(gt_data_dict + mha_list[0])

        gt_rescaled = spatial_normalize.rescale_to_standard(ground_truth, resolution)
        ct_rescaled = spatial_normalize.rescale_to_standard(ct_data, resolution)

        Functions.save_np_array('/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/rescaled_array/',
                                patient_id + '_' + time + '.npy', ct_rescaled, compress=False)
        Functions.save_np_array('/home/zhoul0a/Desktop/其它肺炎/Immune_Pneumonia/rescaled_gt/',
                                patient_id + '_' + time + '.npz', ct_rescaled, compress=True)
        center_z = Functions.center_loc(gt_rescaled, (2,))[0]
        print(center_z)
        Functions.array_stat(gt_rescaled)
        Functions.image_save(np.clip(ct_rescaled[:, :, 256] + 0.5, 0, 1), data_visualize_dict + patient_id + '_' + time
                             , high_resolution=True, gray=True)
        Functions.merge_image_with_mask(np.clip(ct_rescaled[:, :, center_z] + 0.5, 0, 1),
                                        gt_rescaled[:, :, center_z],
                                        save_path=gt_visualize_dict + patient_id + '_' + time, show=False)
