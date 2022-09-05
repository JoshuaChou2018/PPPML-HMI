import numpy as np
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.highlight_semantics as highlight
import visualization.visualize_3d.visualize_stl as stl
import prediction.predict_rescaled as predictor
import analysis.cal_branching_map as branch

array = np.load('/home/zhoul0a/Desktop/open_access/branching.npy')
Functions.image_show(array[:, :, 256])
exit()
rescaled_array = np.load('/home/zhoul0a/Desktop/normal_people/rescaled_ct_array/Scanner-B/Scanner-B_B11.npy')
lung = predictor.predict_lung_masks_rescaled_array(rescaled_array)
blood = predictor.get_prediction_blood_vessel(rescaled_array, lung_mask=lung)

branching_array = branch.calculate_branching_map(lung, blood)

Functions.save_np_array('/home/zhoul0a/Desktop/open_access/', 'branching', branching_array)

exit()

z = 250

high_precision = np.load('/home/zhoul0a/Desktop/NMI_revision_data/visualization/Extended_data_temp/high_precision.npy')

high_recall = np.load('/home/zhoul0a/Desktop/NMI_revision_data/visualization/Extended_data_temp/high_recall.npy')

airway = np.load('/home/zhoul0a/Desktop/NMI_revision_data/visualization/Extended_data_temp/airway.npy')

rim = Functions.get_rim(airway[:, :, z])

image_a = np.zeros([512, 512, 3], 'float32')
image_b = np.zeros([512, 512, 3], 'float32')

image_a[:, :, 0] = high_precision[:, :, z]
image_a[:, :, 1] = high_precision[:, :, z]
image_a[:, :, 2] = high_precision[:, :, z]

image_b[:, :, 0] = high_recall[:, :, z]
image_b[:, :, 1] = high_recall[:, :, z]
image_b[:, :, 2] = high_recall[:, :, z]

image_a[:, :, 0] = image_a[:, :, 0] + rim
image_a[:, :, 1] = image_a[:, :, 1] - rim
image_a[:, :, 2] = image_a[:, :, 2] - rim
image_a = np.clip(image_a, 0, 1)

image_b[:, :, 0] = image_b[:, :, 0] + rim
image_b[:, :, 1] = image_b[:, :, 1] - rim
image_b[:, :, 2] = image_b[:, :, 2] - rim
image_b = np.clip(image_b, 0, 1)

Functions.image_save(image_a, '/home/zhoul0a/Desktop/transfer/17_high_precision.png', high_resolution=True)
Functions.image_save(image_b, '/home/zhoul0a/Desktop/transfer/18_high_recall.png', high_resolution=True)
