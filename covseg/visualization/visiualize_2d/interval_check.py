import Tool_Functions.Functions as Functions
import os
import numpy as np


def slicing_visualize(array_3d, save_dict, interval=5, axis=2):
    """

    :param array_3d:
    :param save_dict:
    :param interval:
    :param axis:
    :return:
    """
    shape = np.shape(array_3d)
    if axis == 2:
        for z in range(0, shape[2], interval):
            if os.path.exists(os.path.join(save_dict, str(z)) + '.png'):
                continue
            image = array_3d[:, :, z]
            Functions.image_save(image, os.path.join(save_dict, str(z)), gray=True, high_resolution=True)
    if axis == 1:
        for y in range(0, shape[1], interval):
            if os.path.exists(os.path.join(save_dict, str(y)) + '.png'):
                continue
            image = array_3d[:, y, :]
            Functions.image_save(image, os.path.join(save_dict, str(y)), gray=True, high_resolution=True)
    if axis == 0:
        for x in range(0, shape[0], interval):
            if os.path.exists(os.path.join(save_dict, str(x)) + '.png'):
                continue
            image = array_3d[x, :, :]
            Functions.image_save(image, os.path.join(save_dict, str(x)), gray=True, high_resolution=True)


def slicing_visualize_pipeline(top_dict_array, top_dict_save, interval=5, axis=2):
    array_name_list = os.listdir(top_dict_array)
    num_arrays = len(array_name_list)
    processed_count = 0
    for name in array_name_list:
        print('processing', name)
        print(num_arrays - processed_count, 'left')
        if name[-1] == 'z':
            array_3d = np.load(os.path.join(top_dict_array, name))['array']
        else:
            array_3d = np.load(os.path.join(top_dict_array, name))
        save_dict = os.path.join(top_dict_save, name[:-4])
        slicing_visualize(array_3d, save_dict, interval=interval, axis=axis)
        processed_count += 1


def slicing_visualize_combine(array_3d_left, array_3d_right, func_merge, save_dict, interval=5, axis=2):

    shape = np.shape(array_3d_left)
    if axis == 2:
        for z in range(0, shape[2], interval):
            if os.path.exists(os.path.join(save_dict, str(z)) + '.png'):
                continue
            image_left = array_3d_left[:, :, z]
            image_right = array_3d_right[:, :, z]

            if np.std(image_left) < 0.0001 or np.std(image_right) < 0.0001:
                continue

            image = func_merge(image_left, image_right)
            Functions.image_save(image, os.path.join(save_dict, str(z)), gray=True, high_resolution=True)
    if axis == 1:
        for y in range(0, shape[1], interval):
            if os.path.exists(os.path.join(save_dict, str(y)) + '.png'):
                continue
            image_left = array_3d_left[:, y, :]
            image_right = array_3d_right[:, y, :]

            if np.std(image_left) < 0.0001 or np.std(image_right) < 0.0001:
                continue

            image = func_merge(image_left, image_right)
            Functions.image_save(image, os.path.join(save_dict, str(y)), gray=True, high_resolution=True)
    if axis == 0:
        for x in range(0, shape[0], interval):
            if os.path.exists(os.path.join(save_dict, str(x)) + '.png'):
                continue
            image_left = array_3d_left[x, :, :]
            image_right = array_3d_right[x, :, :]

            if np.std(image_left) < 0.0001 or np.std(image_right) < 0.0001:
                continue

            image = func_merge(image_left, image_right)
            Functions.image_save(image, os.path.join(save_dict, str(x)), gray=True, high_resolution=True)


def slicing_visualize_combine_pipeline(top_dict_array_left, top_dict_array_right, func_merge,
                                       top_dict_save, interval=5, axis=2):
    array_name_list = os.listdir(top_dict_array_left)
    num_arrays = len(array_name_list)
    processed_count = 0
    for name in array_name_list:
        print('processing', name)
        print(num_arrays - processed_count, 'left')
        if name[-1] == 'z':
            array_3d_left = np.load(os.path.join(top_dict_array_left, name))['array']
        else:
            array_3d_left = np.load(os.path.join(top_dict_array_left, name))

        if os.path.exists(os.path.join(top_dict_array_right, name[:-4] + '.npz')):
            array_3d_right = np.load(os.path.join(top_dict_array_right, name[:-4] + '.npz'))['array']
        else:
            array_3d_right = np.load(os.path.join(top_dict_array_right, name[:-4] + '.npy'))

        save_dict = os.path.join(top_dict_save, name[:-4])
        slicing_visualize_combine(array_3d_left, array_3d_right, func_merge, save_dict, interval=interval, axis=axis)
        processed_count += 1


def combine_rescaled_and_enhanced(rescaled, enhanced):
    rescaled = np.clip(rescaled + 0.5, 0, 1)
    rescaled[0, 0] = 0
    rescaled[-1, -1] = 1
    enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 0.000001)
    shape = np.shape(rescaled)
    image = np.zeros([shape[0], shape[1] * 2], 'float32')
    image[:, 0: shape[1]] = rescaled
    image[:, shape[1]::] = enhanced
    return image


if __name__ == '__main__':
    array_top_dict_left = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/9肺癌/'
    array_top_dict_right = '/home/zhoul0a/Desktop/NMI_revision_data/enhanced_arrays/9肺癌/'
    save_image_top_dict = '/home/zhoul0a/Desktop/NMI_revision_data/visualization/9肺癌/'
    slicing_visualize_combine_pipeline(array_top_dict_left, array_top_dict_right, combine_rescaled_and_enhanced,
                                       save_image_top_dict)

    array_top_dict_left = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/immune_pneumonia/'
    array_top_dict_right = '/home/zhoul0a/Desktop/NMI_revision_data/enhanced_arrays/immune_pneumonia/'
    save_image_top_dict = '/home/zhoul0a/Desktop/NMI_revision_data/visualization/immune_pneumonia/'
    slicing_visualize_combine_pipeline(array_top_dict_left, array_top_dict_right, combine_rescaled_and_enhanced,
                                       save_image_top_dict)

    array_top_dict_left = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/pulmonary_nodules/'
    array_top_dict_right = '/home/zhoul0a/Desktop/NMI_revision_data/enhanced_arrays/pulmonary_nodules/'
    save_image_top_dict = '/home/zhoul0a/Desktop/NMI_revision_data/visualization/pulmonary_nodules/'
    slicing_visualize_combine_pipeline(array_top_dict_left, array_top_dict_right, combine_rescaled_and_enhanced,
                                       save_image_top_dict)

    array_top_dict_left = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/new_follow_up/'
    array_top_dict_right = '/home/zhoul0a/Desktop/NMI_revision_data/enhanced_arrays/new_follow_up/'
    save_image_top_dict = '/home/zhoul0a/Desktop/NMI_revision_data/visualization/new_follow_up/'
    slicing_visualize_combine_pipeline(array_top_dict_left, array_top_dict_right, combine_rescaled_and_enhanced,
                                       save_image_top_dict)
