import numpy as np
import Tool_Functions.Functions as Functions
from analysis.connected_region2d_and_scale_free_stat import get_rim
from prediction.predict_rescaled import get_top_rated_points_use_lung_as_anchor

enhanced = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/progonosis_enhanced/xghf-25_2020-07-29.npz')['array']
enhanced_image = np.clip(enhanced[:, :, 308], 0, 0.1) * 10

gt = np.load('/home/zhoul0a/Desktop/prognosis_project/visible&invisible_lesions/invisible_lesion/xghf-25_2020-07-29.npz')['array']
lung_mask = np.load('/home/zhoul0a/Desktop/prognosis_project/visible&invisible_lesions/lung_mask/xghf-25_2020-07-29.npz')['array']
airway_mask = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/air_way_mask_stage_two/xghf-25/xghf-25_2020-07-29_mask.npz')['array']
blood_mask = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/blood_vessel_mask_stage_two/xghf-25/xghf-25_2020-07-29_mask.npz')['array']
blood_image = blood_mask[:, :, 308]
airway_image = airway_mask[:, :, 308]
lung_image = lung_mask[:, :, 308]

gt_ratio = np.sum(gt) / np.sum(lung_mask) * 1.1

gt_restrain = np.array(gt[:, :, 308])
for i in range(20):
    gt_rim = get_rim(gt_restrain, outer=True)
    gt_restrain = gt_restrain + gt_rim

"""
gt_image = gt[:, :, 308]

for i in range(2):
    gt_rim = get_rim(gt_image, outer=True)
    gt_image = gt_image + gt_rim
gt_image = gt_image * lung_image * (1 - airway_image) * (1 - blood_image)
Functions.merge_image_with_mask(enhanced_image, gt_image, save_path='/home/zhoul0a/Desktop/transfer/04_gt.png', high_resolution=True)
"""

probability_map_mpu_normal = Functions.load_nii('/home/zhoul0a/Desktop/transfer/MPUnet-J/normal/raw_predictions/nii_files/xghf-25_2020-07-29_PRED.nii.gz')[:, :, :, 1]
mask_normal_mpu = get_top_rated_points_use_lung_as_anchor(lung_mask, probability_map_mpu_normal, gt_ratio / 2)
image_normal_mpu = mask_normal_mpu[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image)
print(Functions.f1_sore_for_binary_mask(mask_normal_mpu, gt))
Functions.merge_image_with_mask(enhanced_image, image_normal_mpu, save_path='/home/zhoul0a/Desktop/transfer/05_mpu_normal.png', high_resolution=True)


probability_map_mpu_enhanced = Functions.load_nii('/home/zhoul0a/Desktop/transfer/MPUnet-J/enhanced/raw_predictions/nii_files/xghf-25_2020-07-29_PRED.nii.gz')[:, :, :, 1]
mask_enhanced_mpu = get_top_rated_points_use_lung_as_anchor(lung_mask, probability_map_mpu_enhanced, gt_ratio)
image_normal_mpu = mask_enhanced_mpu[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image) * gt_restrain
print(Functions.f1_sore_for_binary_mask(mask_enhanced_mpu, gt))
Functions.merge_image_with_mask(enhanced_image, image_normal_mpu, save_path='/home/zhoul0a/Desktop/transfer/06_mpu_enhanced.png', high_resolution=True)


u3d_prob_normal = np.load('/home/zhoul0a/Desktop/transfer/MPUnet-J/3DUNet_ori_seg_test/xghf-25_2020-07-29_predictions.h5.npy')
mask_normal_u3d = get_top_rated_points_use_lung_as_anchor(lung_mask, u3d_prob_normal, gt_ratio / 2)
image_normal_u3d = mask_normal_u3d[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image)
print(Functions.f1_sore_for_binary_mask(mask_normal_u3d, gt))
Functions.merge_image_with_mask(enhanced_image, image_normal_u3d, save_path='/home/zhoul0a/Desktop/transfer/07_3du_normal.png', high_resolution=True)


u3d_prob_enhanced = np.load('/home/zhoul0a/Desktop/transfer/MPUnet-J/3DUNet_enhanced_test_predict/xghf-25_2020-07-29_predictions.h5.npy')
mask_enhance_u3d = get_top_rated_points_use_lung_as_anchor(lung_mask, u3d_prob_enhanced, gt_ratio)
image_enhance_u3d = mask_enhance_u3d[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image) * gt_restrain
print(Functions.f1_sore_for_binary_mask(mask_enhance_u3d, gt))
Functions.merge_image_with_mask(enhanced_image, image_enhance_u3d, save_path='/home/zhoul0a/Desktop/transfer/08_3du_enhanced.png', high_resolution=True)


ours_prob = np.load('/home/zhoul0a/Desktop/transfer/MPUnet-J/ours.npy')

prob_normal = ours_prob + u3d_prob_normal * u3d_prob_normal
mask_normal_ours = get_top_rated_points_use_lung_as_anchor(lung_mask, prob_normal, gt_ratio / 2)
image_normal_ours = mask_normal_ours[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image)
print(Functions.f1_sore_for_binary_mask(mask_normal_ours, gt))
Functions.merge_image_with_mask(enhanced_image, image_normal_ours, save_path='/home/zhoul0a/Desktop/transfer/09_ours_normal.png', high_resolution=True)


prob_enhanced = ours_prob * ours_prob + u3d_prob_enhanced * u3d_prob_enhanced * u3d_prob_enhanced
mask_enhanced_ours = get_top_rated_points_use_lung_as_anchor(lung_mask, prob_enhanced, gt_ratio)
image_enhanced_ours = mask_enhanced_ours[:, :, 308] * lung_image * (1 - airway_image) * (1 - blood_image) * gt_restrain
print(Functions.f1_sore_for_binary_mask(mask_enhanced_ours, gt))
Functions.merge_image_with_mask(enhanced_image, image_enhanced_ours, save_path='/home/zhoul0a/Desktop/transfer/10_ours_enhance.png', high_resolution=True)
exit()
exit()
exit()

mpu_predict_enhance = Functions.load_nii('/home/zhoul0a/Desktop/transfer/MPUnet-J/enhanced/predictions/nii_files/xghf-25_2020-07-29_PRED.nii.gz')

Functions.merge_image_with_mask(enhanced_image, mpu_predict_enhance[:, :, 308])

print(Functions.f1_sore_for_binary_mask(mpu_predict_enhance, gt))
exit()
