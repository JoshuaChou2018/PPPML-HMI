import numpy as np
import Tool_Functions.Functions as Functions
import os

file_name_list = os.listdir('/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/new_follow_up/')

for name in file_name_list:
    array = np.load('/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/new_follow_up/' + name)

    Functions.save_np_array('/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/rescaled_ct_compressed/', name[:-4], array, compress=True)
