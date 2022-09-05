import os
import numpy as np
import Tool_Functions.Functions as Functions
import format_convert.dcm_np_converter as convert
import post_processing.parenchyma_enhancement as enhancement

top_dict_rescaled = '/home/zhoul0a/Desktop/NMI_revision_data/rescaled_ct/immune_pneumonia/'
top_dict_mask = '/home/zhoul0a/Desktop/NMI_revision_data/masks/immune_pneumonia/'
save_dict = '/home/zhoul0a/Desktop/NMI_revision_data/enhanced_arrays/immune_pneumonia/'


enhancement.enhance_pipeline(top_dict_rescaled, top_dict_mask, save_dict, enhance_std_interval=(-1, 7))
