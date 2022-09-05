import sample_manager.sample_slicer_multi_classes as multi_slicer
import sample_manager.loss_weight_voxel_wise as loss_weight_cal


def stage_one_training_samples(rescaled_array_dict, training_sample_save_dict, data_channel, enhanced_channel,
                               positive_gt_channel_list, window=(-5, -2, 0, 2, 5), neglect_interval=None):
    """
    this prepare training samples for stage one training.
    rescaled_array with shape: [512, 512, 512, data_channel + enhanced_channel + semantic_channel]
    pre-requisite: rescaled arrays. details about rescaled arrays please see sample_manager.sample_slicer_multi_classes
    :param window:
    :param positive_gt_channel_list: which gt channels are considered as positive, e.g. two gt channels, the first is
    positive, then this is (0,)
    :param enhanced_channel: freq int, the number of enhanced channels
    :param data_channel: freq int, the number of data channels
    :param neglect_interval: freq int, slice_id % neglect_interval == 0 will be use as blank sample (no positive pixel)
    None means all samples contains positive pixel
    :param rescaled_array_dict: into this dict, are arrays described in sample_manager.sample_slicer_multi_classes
    :param training_sample_save_dict: where to save training samples
    :return: None
    """
    # data_channel is always 1 for CT. enhance channel is 0 for stage one training.

    # if there is only one semantic channel, then the positive semantic is (0,), only in this case can you not include
    # the negatives, otherwise, you must include all semantics.
    # e.g. the first semantic channel is parenchyma, second is nodule, third is infection, fourth is tumor, then,
    # positive_semantic_channel = (1, 2, 3). You cannot say: positive is (0, 1, 2) and neglect the parenchyma semantic.

    # infection, lung, window is (-5, -2, 0, 2, 5); tracheae and airways vessel (-1, 0, 1)
    multi_slicer.prepare_training_set(rescaled_array_dict, training_sample_save_dict, data_channel, enhanced_channel,
                                      positive_gt_channel_list, window=window, neglect_interval=neglect_interval)


def stage_one_training_samples_v2(rescaled_array_dict, dict_for_rescaled_gt, gt_channel, training_sample_save_dict,
                                  window=(-1, 0, 1), force_select_interval=10):
    """
    this prepare training samples, for only one positive tissue segmentation
    pre-requisite: rescaled arrays. details about rescaled arrays please see sample_manager.sample_slicer_multi_classes
    :param force_select_interval: if slice_id % neglect_interval == 0, it will always be included, otherwise, the
    slice must contain some positive points.
    :param window:
    :param dict_for_rescaled_gt: should be in shape [512, 512, 512, semantics]
    :param gt_channel: freq int, which semantic to predict
    :param rescaled_array_dict: into this dict, are arrays described in sample_manager.sample_slicer_multi_classes
    :param training_sample_save_dict: where to save training samples
    :return: None
    """
    # data_channel is always 1 for CT. enhance channel is 0 for stage one training.

    # if there is only one semantic channel, then the positive semantic is (0,), only in this case can you not include
    # the negatives, otherwise, you must include all semantics.
    # e.g. the first semantic channel is parenchyma, second is nodule, third is infection, fourth is tumor, then,
    # positive_semantic_channel = (1, 2, 3). You cannot say: positive is (0, 1, 2) and neglect the parenchyma semantic.

    # infection, lung, window is (-5, -2, 0, 2, 5); tracheae and airways vessel (-1, 0, 1)
    multi_slicer.prepare_training_set_v2(rescaled_array_dict, dict_for_rescaled_gt, gt_channel,
                                         training_sample_save_dict, 1, 0, (0,), window=window,
                                         neglect_interval=force_select_interval)


def stage_two_training_samples(enhanced_array_dict, training_sample_save_dict):
    """
    this prepare training samples for stage one training.
    pre-requisite: enhanced arrays. details about enhanced arrays please see sample_manager.sample_slicer_multi_classes
    :param enhanced_array_dict: into this dict, are arrays described in sample_manager.sample_slicer_multi_classes
    :param training_sample_save_dict: where to save training samples
    :return: None
    """
    # data_channel is always 1 for CT. enhance channel is 0 for stage one training.

    # if there is only one semantic channel, then the positive semantic is (0,), only in this case can you not include
    # the negatives, otherwise, you must include all semantics.
    # e.g. the first semantic channel is parenchyma, second is nodule, third is infection, fourth is tumor, then,
    # positive_semantic_channel = (1, 2, 3). You cannot say: positive is (0, 1, 2) and neglect the parenchyma semantic.

    # tracheae and airways vessel (-1, 0, 1)
    multi_slicer.prepare_training_set(enhanced_array_dict, training_sample_save_dict, 1, 2, (0,),
                                      window=(-1, 0, 1))


def cal_balance_weight_array(root_dict, sample_file_name, balance_array_file_name):
    """
    :param root_dict:
    :param sample_file_name: training_sample_save_dict in stage_one_training_samples is
    os.path.join(root_dict, sample_file_name)
    :param balance_array_file_name: each training sample will have freq balance weight array, the directory structure is
    the same with training samples, and "balance_array_file_name" corresponds to "sample_file_name"
    :return:
    """
    loss_weight_cal.prepare_weight_balance(root_dict, sample_file_name, balance_array_file_name)


if __name__ == "__main__":
    stage_one_training_samples('/home/zhoul0a/Desktop/vein_artery_identification/rescaled_array_root_2/',
                               '/home/zhoul0a/Desktop/vein_artery_identification/segment_root/training_samples/',
                               1, 2, (1, 2), window=(-5, -2, 0, 2, 5), neglect_interval=10)
    exit()
    stage_one_training_samples_v2('/home/zhoul0a/Desktop/vein_artery_identification/rescaled_ct/',
                                  '/home/zhoul0a/Desktop/vein_artery_identification/rescaled_gt/', 0,
                                  '/home/zhoul0a/Desktop/vein_artery_identification/segment_heart/training_samples/')

    exit()
    stage_two_training_samples('/home/zhoul0a/Desktop/blood_vessel_seg_non-enhance/enhanced_arrays/',
                               '/home/zhoul0a/Desktop/blood_vessel_seg_non-enhance/training_sample_stage_two/')
