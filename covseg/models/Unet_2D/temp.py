import numpy as np
import Tool_Functions.Functions as Functions
import os
import pre_processing.read_in_CT as load_dcm
import visualization.visualize_stl as vl
import visualization.slice_by_slice_check as slice_vi

diction = '/home/zhoul0a/Desktop/Lung_CAD_NMI/raw_data/blood_vessel/'
patient_id_list = os.listdir(diction)
for patient in patient_id_list:
    if patient == 'xwqg-A000131':
        continue
    second_dict = diction + patient + '/'
    list_file = os.listdir(second_dict)
    for file in list_file:
        if file == 'stl':
            continue
        third_dict = second_dict + file + '/Data/raw_data/'
        dcm_names = os.listdir(third_dict)
        print(patient)
        print(Functions.wc_ww(third_dict + dcm_names[0]))
        #Functions.array_stat()


def test():
    array = load_dcm.stack_dcm_files('/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/tracheae_seg/patients/Normal/A11/', wc_ww=(-600, 1600))[0]
    array = np.clip(array, -0.5, 0.5) + 0.5
    print(np.shape(array)[2])
    num = np.shape(array)[2]
    for i in range(num):
        Functions.image_save(array[:, :, i], '/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/images/normal_A11/' + str(i) + '.png', True)


test()

array = vl.convert_mha_to_npy('/home/zhoul0a/Desktop/Lung_CAD_NMI/raw_data/blood_vessel/xwqg-A000040/2019-05-23/Data/ground_truth/pa(分割).mha')

images = load_dcm.stack_dcm_files('/home/zhoul0a/Desktop/Lung_CAD_NMI/raw_data/blood_vessel/xwqg-A000040/2019-05-23/Data/raw_data/', wc_ww=(60, 400))[0]

images += 0.5

slice_vi.visualize_mask_and_raw_array(array, images, '/home/zhoul0a/Desktop/Lung_CAD_NMI/applications/images/xwqg-A000040_high_contrast/', clip=True)

