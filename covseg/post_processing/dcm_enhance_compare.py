
"""
this file generate the four grid image for doctor Xie to determine the abnormal regions
you should first predict the enhanced rescaled
"""
import Tool_Functions.Functions as Functions
import Tool_Functions.id_time_generator as generator
import numpy as np
import pre_processing.spatial_normalize as normalize
import visualization.visualize_stl as visualize
import os


def process_one_scan(patient, time):

    if os.path.exists('/home/zhoul0a/Desktop/normal_people/pictures/plot_new_outer/' + patient + '/' + time + '/'):
        print("this scan has been processed")
        return None

    mask_root_dict = '/home/zhoul0a/Desktop/normal_people/rescaled_masks_refined/blood_vessel_mask_stage_two/'
    enhance_root_dict = '/home/zhoul0a/Desktop/normal_people/rescaled_ct_enhanced_upperlobe_outer/'
    dcm_root_dict = '/home/zhoul0a/Desktop/normal_people/raw_data/'

    mask = np.load(mask_root_dict + patient + '/' + patient + '_' + time + '_mask_refine.npz')['array']

    try:
        array = np.load(enhance_root_dict + patient + '_' + time + '.npy')
    except:
        array = np.load(enhance_root_dict + patient + '_' + time + '.npz')['array']


    array_left = array[:, 0:512, :]
    array_right = array[:, 512::, :]
    if os.path.exists(dcm_root_dict + patient + '/' + time + '/Data/raw_data/'):
        dcm_file_dict = dcm_root_dict + patient + '/' + time + '/Data/raw_data/'
    else:
        dcm_file_dict = dcm_root_dict + patient + '/' + time + '/'
    array_left = normalize.rescale_to_original_2(dcm_file_dict, array_left)
    array_right = normalize.rescale_to_original_2(dcm_file_dict, array_right)

    array = np.concatenate([array_left, array_right], axis=1)

    mask = normalize.rescale_to_original_2(dcm_file_dict, mask)
    mask = np.array(mask > 0, 'float32')

    array[:, 512::, :] = array[:, 512::, :] - mask
    array = np.clip(array, 0, 0.2) * 5

    dcm_fn_list = os.listdir(dcm_file_dict)
    num_images = len(dcm_fn_list)

    save_mha_dict = '/home/zhoul0a/Desktop/normal_people/pictures/pictures/mha_and_stl_original/' + patient + '/' + time + '/'
    if not os.path.exists(save_mha_dict + 'blood_vessel_mask.mha'):
        visualize.save_numpy_as_stl(mask, save_mha_dict, 'blood_vessel_mask')

    save_dict = '/home/zhoul0a/Desktop/normal_people/pictures/plot_new_outer/' + patient + '/' + time + '/'
    for fn in dcm_fn_list:
        path = dcm_file_dict + fn
        image, slice_id = Functions.load_dicom(path)
        up = Functions.dicom_and_prediction(mask[:, :, num_images - slice_id], image)
        down = np.zeros((512, 1024, 3))
        down[:, :, 0] = array[:, :, num_images - slice_id]
        down[:, :, 1] = array[:, :, num_images - slice_id]
        down[:, :, 2] = array[:, :, num_images - slice_id]
        merge = np.concatenate((up, down), axis=0)
        Functions.image_save(merge, save_dict + str(slice_id), gray=False, high_resolution=True)


info_list = generator.return_all_tuples_for_rescaled_ct('/home/zhoul0a/Desktop/normal_people/rescaled_ct_array/')
processed = 0

for patient_id, scan_time in info_list[::3]:
    if scan_time in ['B37']:
        continue

    if patient_id in []:
        continue
    print('\n\n\n', patient_id, scan_time, processed, '\n\n\n')
    process_one_scan(patient_id, scan_time)
    processed += 1




