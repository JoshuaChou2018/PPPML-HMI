import format_convert.dcm_np_converter as converter
import numpy as np
import Tool_Functions.Functions as Functions

array_0 = converter.dcm_to_spatial_rescaled('/home/zhoul0a/Desktop/KFSH/breast_cancer/raw_data/2365203-20211109T154152Z-001/2365203/DCE-MRI/ser013-T1_fl3d_tra_dynaVIEWS_spair_8+1/',
                                            target_shape=(464, 464, 240), target_resolution=(0.75, 0.75, 0.75), tissue='breast')

print(np.shape(array_0))

array_2 = converter.dcm_to_spatial_rescaled('/home/zhoul0a/Desktop/KFSH/breast_cancer/raw_data/2365203-20211109T154152Z-001/2365203/DCE-MRI/ser021- T1_fl3d_tra_dynaVIEWS_spair_8+1/',
                                            target_shape=(464, 464, 240), target_resolution=(0.75, 0.75, 0.75), tissue='breast')

print(np.shape(array_2))

array_4 = converter.dcm_to_spatial_rescaled('/home/zhoul0a/Desktop/KFSH/breast_cancer/raw_data/2365203-20211109T154152Z-001/2365203/DCE-MRI/ser029- T1_fl3d_tra_dynaVIEWS_spair_8+1/',
                                            target_shape=(464, 464, 240), target_resolution=(0.75, 0.75, 0.75), tissue='breast')

print(np.shape(array_4))


def min_max_normalize(input_array, new_max, new_min):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    interval = new_max - new_min
    assert interval > 0
    return (input_array - min_value) / (max_value - min_value) * interval + new_min


channel_0 = min_max_normalize(array_2, 1, 0)
channel_1 = min_max_normalize(array_2 - array_0, 1, -1)
channel_2 = min_max_normalize(-array_4 + array_2, 1, -1)

rescaled_array = np.zeros((464, 464, 240, 3), 'float32')
rescaled_array[:, :, :, 0] = channel_0  # data channel
rescaled_array[:, :, :, 1] = channel_1  # enhanced channel
rescaled_array[:, :, :, 2] = channel_2  # enhanced channel


import prediction.predict_rescaled as predictor

mask = predictor.predict_breast_tumor_dcm_mri(rescaled_array)

max_prob = np.max(mask)

print("max probability:", max_prob)

mask = mask / max_prob

z_mid = int(np.mean(np.where(mask > 0.9)[2]))

Functions.image_show(mask[:, :, z_mid])

Functions.merge_image_with_mask(rescaled_array[:, :, z_mid, 0], mask[:, :, z_mid], save_path='/home/zhoul0a/Desktop/KFSH/breast_cancer/visualize/2365203.png')
