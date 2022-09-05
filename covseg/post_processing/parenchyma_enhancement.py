"""
input: rescaled_ct scan with shape [512, 512, 512]
output: removed airway, blood vessel and set the parenchyma value to 0,
"""
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.visualize_stl as visualize
import os
import numpy as np
import post_processing.remove_airway_blood_vessel as extend_functions
import prediction.predict_rescaled as predictor
import random
import warnings
np.set_printoptions(threshold=np.inf)


def remove_airway_and_blood_vessel_based_on_upper_frontal(rescaled_ct, lung_mask=None, airway=None, blood_vessel=None,
                                                          extend_ratio=1.1, max_diameter=50, show_stl=False, parenchyma_value=False):
    """
    :param rescaled_ct: the [512, 512, 512] spatial and signal normalized data
    :param lung_mask: we will predict if it is None
    :param airway: the airway mask, we will predict if it is None
    :param blood_vessel: the blood_vessel mask, we will predict if it is None
    :param extend_ratio: the diameter of this region will be extend by this ratio and this ratio + 0.1. The tissue in
    the middle of these two regions is ASSUMED AS PARENCHYMA!
    :param max_diameter: if the diameter of the region is greater than the max_diameter, we extend the region to new
    diameter: (extend_ratio - 1) * max_diameter + old_diameter
    :param show_stl: whether to visualize the 3D model after segmentation
    :param parenchyma_value:  False, not return it, True, return it
    :return: enhanced_array in shape [512, 512, 512], which is enhanced rescaled ct images;
    or enhanced_array, parenchyma_value
    """
    if lung_mask is None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct)
    lung_mask_box = Functions.get_bounding_box(lung_mask)
    print("lung_mask_box:", lung_mask_box)
    lung_length_z = lung_mask_box[2][1] - lung_mask_box[2][0]
    superior_start = int(lung_mask_box[2][1] - lung_length_z * 0.32988803223955687)  # this is the upper
    superior_end = lung_mask_box[2][1]
    air_way_merge_z_bounding_box = Functions.get_bounding_box(lung_mask[:, :, superior_start])
    upper_start = air_way_merge_z_bounding_box[0][0]  # this is the frontal
    upper_end = air_way_merge_z_bounding_box[0][1] - int((air_way_merge_z_bounding_box[0][1] -
                                                          air_way_merge_z_bounding_box[0][0])/2)
    print("lung length on z:", lung_length_z, "superior range:", superior_start, superior_end,
          "upper range:", upper_start, upper_end)
    upper_superior_mask = np.zeros(np.shape(lung_mask), 'float32')
    upper_superior_mask[upper_start: upper_end, :, superior_start: superior_end] = 1.0
    upper_superior_mask = upper_superior_mask * lung_mask  # this is the mask for upper frontal lung

    if airway is None:
        refined_airway_mask = predictor.get_prediction_airway(rescaled_ct, lung_mask=lung_mask)
    else:
        refined_airway_mask = airway
    if blood_vessel is None:
        refined_blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct, lung_mask=lung_mask)
    else:
        refined_blood_vessel_mask = blood_vessel

    if show_stl:
        print("lung")
        visualize.visualize_numpy_as_stl(lung_mask)
        print("airway")
        visualize.visualize_numpy_as_stl(refined_airway_mask)
        print("blood vessels")
        visualize.visualize_numpy_as_stl(refined_blood_vessel_mask)

    rescaled_ct = rescaled_ct * lung_mask
    visible_non_infection = np.array((refined_blood_vessel_mask + refined_airway_mask) * lung_mask > 0.5, 'float32')
    rescaled_ct_original = np.array(rescaled_ct)

    assert extend_ratio > 1
    print("extending air way")
    visible_extended_outer = extend_functions.extend_tubes(visible_non_infection, None, extend_ratio + 0.1, int(max_diameter * 1.1))
    visible_extended = extend_functions.extend_tubes(visible_non_infection, None, extend_ratio, max_diameter)

    visible_extended = visible_extended_outer - visible_extended
    context_mask = visible_extended - visible_non_infection

    context_mask = np.clip(context_mask, 0, 1)

    context_mask = context_mask * upper_superior_mask

    print(Functions.stat_on_mask(rescaled_ct_original, context_mask))

    num_context_points = np.sum(context_mask)
    print("there are:", num_context_points, "parenchyma sample points")
    if num_context_points < 10000:
        warnings.warn('Too less (<10000) sampled parenchyma points. Maybe use "general sampling"', SyntaxWarning)
    context = context_mask * rescaled_ct_original + context_mask * 10
    context = np.reshape(context, (-1,))
    context = np.sort(context)
    total_points = len(context)

    percentile = 50
    threshold = context[total_points - int(num_context_points * (100 - percentile) / 100)] - 10

    if threshold > -0.1:
        warnings.warn('Too high threshold. Maybe use "general sampling"?', SyntaxWarning)

    print("the context is:", threshold, 'at percentile', percentile)

    rescaled_ct[np.where(visible_non_infection >= 0.5)] = threshold  # removed the airway and blood vessel
    rescaled_ct = rescaled_ct - threshold * lung_mask  # threshold is the value of lung parenchyma
    # set the parenchyma value to zero

    enhanced_array = np.zeros([512, 512, 512], 'float32')
    enhanced_array[:, :, :] = rescaled_ct
    enhanced_array = np.clip(enhanced_array, -0.05, 0.25)

    if parenchyma_value:
        return enhanced_array, threshold

    return enhanced_array


def remove_airway_and_blood_vessel_general_sampling(rescaled_ct, lung_mask=None, airway=None, blood_vessel=None,
                                                    extend_ratio=1.1, max_diameter=50, show_stl=False, parenchyma_value=False, window=False):
    """
    :param window: if True, return the scan optimal window
    :param rescaled_ct: the [512, 512, 512] spatial and signal normalized data
    :param lung_mask: we will predict if it is None
    :param airway: the airway mask, we will predict if it is None
    :param blood_vessel: the blood_vessel mask, we will predict if it is None
    :param extend_ratio: the diameter of this region will be extend by this ratio. And the extended region is ASSUMED AS
    PARENCHYMA!
    :param max_diameter: if the diameter of the region is greater than the max_diameter, we extend the region to new
    diameter: (extend_ratio - 1) * max_diameter + old_diameter
    :param show_stl: whether to visualize the 3D model after segmentation
    :param parenchyma_value: False, not return it, True, return it
    :return: enhanced_array in shape [512, 512, 512], which is enhanced rescaled ct images;
    or enhanced_array, parenchyma_value
    """
    if lung_mask is None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct)
    lung_mask_box = Functions.get_bounding_box(lung_mask)
    print("lung_mask_box:", lung_mask_box)

    if airway is None:
        refined_airway_mask = predictor.get_prediction_airway(rescaled_ct, lung_mask=lung_mask)
    else:
        refined_airway_mask = airway
    if blood_vessel is None:
        refined_blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct, lung_mask=lung_mask)
    else:
        refined_blood_vessel_mask = blood_vessel

    if show_stl:
        print("lung")
        visualize.visualize_numpy_as_stl(lung_mask)
        print("airway")
        visualize.visualize_numpy_as_stl(refined_airway_mask)
        print("blood vessels")
        visualize.visualize_numpy_as_stl(refined_blood_vessel_mask)

    rescaled_ct = rescaled_ct * lung_mask
    visible_non_infection = np.array((refined_blood_vessel_mask + refined_airway_mask) * lung_mask > 0.5, 'float32')
    rescaled_ct_original = np.array(rescaled_ct)

    assert extend_ratio > 1
    print("extending air way and blood vessels")
    visible_extended_outer = extend_functions.extend_tubes(visible_non_infection, None, extend_ratio + 0.1,
                                                           int(max_diameter * 1.1))
    visible_extended = extend_functions.extend_tubes(visible_non_infection, None, extend_ratio, max_diameter)

    visible_extended = visible_extended_outer - visible_extended
    context_mask = visible_extended - visible_non_infection

    context_mask = np.clip(context_mask, 0, 1)

    num_context_points = np.sum(context_mask)
    print("there are:", num_context_points, "parenchyma sample points")

    context = context_mask * rescaled_ct_original + context_mask * 10
    context = np.reshape(context, (-1,))
    context = np.sort(context)
    total_points = len(context)

    percentile = 50
    threshold = context[total_points - int(num_context_points * (100 - percentile) / 100)] - 10

    std = np.std(context[total_points - int(num_context_points * 0.9): total_points - int(num_context_points * 0.1)])

    print("the context is:", threshold, 'at percentile', percentile)

    rescaled_ct[np.where(visible_non_infection >= 0.5)] = threshold  # removed the airway and blood vessel
    rescaled_ct = rescaled_ct - threshold * lung_mask  # threshold is the value of lung parenchyma
    # set the parenchyma value to zero

    enhanced_array = np.zeros([512, 512, 512], 'float32')
    enhanced_array[:, :, :] = rescaled_ct
    enhanced_array = np.clip(enhanced_array, -0.05, 0.25)

    if parenchyma_value:
        return enhanced_array, threshold

    if window:
        return enhanced_array, -600 + threshold * 1600, std * 1.5 * 1600

    return enhanced_array


def enhance_with_quantile(rescaled_ct, lung_mask=None, airway=None, blood_vessel=None, known_abnormal_mask=None,
                          quantile_normal=50, quantile_for_std=(25, 75), enhance_std_interval=(0, 10)):
    """

    :param rescaled_ct:
    :param lung_mask:
    :param airway:
    :param blood_vessel:
    :param known_abnormal_mask:
    :param quantile_normal: remove airways, blood vessels and known abnormal, sort the signals from small to high.
    quantile is from 0 to 100, to determine the signal for normal parenchyma.
    :param quantile_for_std: we calculate the std for normal tissue
    :param enhance_std_interval: the window width for showing the data, the enhanced array will be shown in
    np.clip(array slice, std * enhanced_std_interval[0], std * enhanced_std_interval[1])
    :return: np.clip(array, std * enhanced_std_interval[0], std * enhanced_std_interval[1])
    """
    if lung_mask is None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct)
    lung_mask_box = Functions.get_bounding_box(lung_mask)
    print("lung_mask_box:", lung_mask_box)

    if airway is None:
        airway = predictor.get_prediction_airway(rescaled_ct, lung_mask=lung_mask)

    if blood_vessel is None:
        blood_vessel = predictor.get_prediction_blood_vessel(rescaled_ct, lung_mask=lung_mask)

    sample_region = lung_mask * (1 - airway) * (1 - blood_vessel)
    if known_abnormal_mask is not None:
        sample_region = sample_region * (1 - known_abnormal_mask)

    locations_list = np.where(sample_region > 0.5)
    number_voxel_for_sample_region = len(locations_list[0])

    sample_signal_list = []
    for i in range(40000):
        sample_id = int(random.uniform(0, number_voxel_for_sample_region - 1))
        sample_signal_list.append(rescaled_ct[locations_list[0][sample_id], locations_list[1][sample_id],
                                              locations_list[2][sample_id]])
    sample_signal_list.sort()
    base_parenchyma = sample_signal_list[int(400 * quantile_normal)]
    print("the base signal for normal parenchyma is:", base_parenchyma)
    std_for_normal = np.std(sample_signal_list[int(quantile_for_std[0] * 400): int(quantile_for_std[1] * 400)])
    print("the std for normal is", std_for_normal)

    enhanced_array = np.clip(rescaled_ct, enhance_std_interval[0] * std_for_normal + base_parenchyma,
                             enhance_std_interval[1] * std_for_normal + base_parenchyma) * sample_region

    return enhanced_array + (enhance_std_interval[0] * std_for_normal + base_parenchyma) * (1 - sample_region)


def enhance_pipeline(top_dict_rescaled_ct, top_dict_masks, top_dict_save, quantile_normal=50, quantile_for_std=(25, 75),
                     enhance_std_interval=(0, 10)):
    rescaled_array_name_list = os.listdir(top_dict_rescaled_ct)
    num_arrays = len(rescaled_array_name_list)
    processed_count = 0
    for name in rescaled_array_name_list:
        print("processing", name)
        if os.path.exists(top_dict_save + name[:-4] + '.npz'):
            processed_count += 1
            print("processed", num_arrays - processed_count, 'left')
            continue
        rescaled_array = np.load(top_dict_rescaled_ct + name)
        lung = np.load(top_dict_masks + 'lung/' + name[:-4] + '.npz')['array']
        airway = np.load(top_dict_masks + 'airway/' + name[:-4] + '.npz')['array']
        blood = np.load(top_dict_masks + 'blood/' + name[:-4] + '.npz')['array']
        enhanced = enhance_with_quantile(rescaled_array, lung, airway, blood, quantile_normal=quantile_normal,
                                         quantile_for_std=quantile_for_std, enhance_std_interval=enhance_std_interval)
        Functions.save_np_array(top_dict_save, name[:-4], enhanced, True)
        processed_count += 1
        print("processed", num_arrays - processed_count, 'left')


def prepare_arrays_raw_for_normal_and_hospitalize(rescaled_ct, lung_mask=None, airway=None, blood_vessel=None,
                                                  lesion=None, extend_ratio=1.1, max_diameter=50, normal=True, mask_name=None, save_dict=None, upperlobe=True):
    """
    :param rescaled_ct: the [512, 512, 512] spatial and signal normalized data
    :param lung_mask: the [512, 512, 512] mask array, we will predict if it is None
    :param airway: the airway mask, we will predict if it is None
    :param blood_vessel: the blood_vessel mask, we will predict if it is None
    :param lesion: the mask of the lesion, we will predict if it is None AND "normal" is False
    :param extend_ratio: the diameter of this region will be extend by this ratio. And the extended region is ASSUMED AS
    PARENCHYMA!
    :param max_diameter: if the diameter of the region is greater than the max_diameter, we extend the region to new
    diameter: (extend_ratio - 1) * max_diameter + old_diameter
    :param normal: if it is true and the lesion mask is None, predict the lesion
    :return: the arrays_raw in shape [512, 512, 512, 2]:
    arrays_raw[:, :, :, 0] is the enhanced_array rescaled ct data, which is the input images
    arrays_raw[:, :, :, 1] is the mask for lesion, which is the ground truth (can be all zeros)
    """
    if lung_mask is None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct)

    if airway is None:
        airway = predictor.get_prediction_airway(rescaled_ct, lung_mask=lung_mask)

    if blood_vessel is None:
        blood_vessel = predictor.get_prediction_blood_vessel(rescaled_ct, lung_mask=lung_mask)
    if lesion is not None:
        assert normal is not True
    if lesion is None and normal is not True:
        lesion = predictor.predict_covid_19_infection_rescaled_array(rescaled_ct, lung_mask=lung_mask)

    if normal is True:
        lesion = np.zeros([512, 512, 512], 'float32')

    if mask_name is not None:
        Functions.save_np_array(os.path.join(save_dict, 'lung_masks') + '/', mask_name, lung_mask, True)
        Functions.save_np_array(os.path.join(save_dict, 'airway_stage_two') + '/', mask_name, airway, True)
        Functions.save_np_array(os.path.join(save_dict, 'blood_vessel_stage_two') + '/', mask_name, blood_vessel, True)
        Functions.save_np_array(os.path.join(save_dict, 'lesion') + '/', mask_name, lesion, True)

    if upperlobe:
        enhanced_rescaled_ct = remove_airway_and_blood_vessel_based_on_upper_frontal(rescaled_ct, lung_mask, airway,
                                                                                 blood_vessel, extend_ratio, max_diameter, False)
    else:
        enhanced_rescaled_ct = remove_airway_and_blood_vessel_general_sampling(rescaled_ct, lung_mask, airway,
                                                                               blood_vessel, extend_ratio, max_diameter, False)
    arrays_raw = np.zeros([512, 512, 512, 2], 'float32')
    arrays_raw[:, :, :, 0] = enhanced_rescaled_ct
    arrays_raw[:, :, :, 1] = lesion

    return arrays_raw


if __name__ == "__main__":
    exit()