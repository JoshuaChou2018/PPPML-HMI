import numpy as np
import Tool_Functions.Functions as Functions
import visualization.visualize_3d.highlight_semantics as highlight
import os

top_dict_ct = '/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_array/'
top_dict_mask = '/home/zhoul0a/Desktop/prognosis_project/masks/hospitalization/'
top_dict_enhanced = '/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_enhanced/hospitalization/'
top_dict_save = '/home/zhoul0a/Desktop/prognosis_project/visualize/supplementary/comparision/'

fn_list = os.listdir('/home/zhoul0a/Desktop/prognosis_project/masks/hospitalization/visible_lesion/')
# ['xghf-40_2020-04-29.npy', 'xghf-4_2020-02-15.npy', 'xghf-10_2020-04-27.npy', 'xghf-12_2020-05-01.npy',
#            'xghf-15_2020-04-29.npy', 'xghf-17_2020-04-29.npy', 'xghf-23_2020-03-09.npy', 'xghf-24_2020-03-10.npy']
for fn in ['xghf-23_2020-03-09.npy']:
    print("processing:", fn)
    """
    if os.path.exists(top_dict_save + fn[:-4] + '_3.png'):
        print("processed")
        continue
    """
    rescaled_ct = np.load(top_dict_ct + fn[:-4] + '.npy')

    airway_one = np.load(top_dict_mask + "airway_stage_one/" + fn[:-4] + '.npz')['array']
    airway_two = np.load(top_dict_mask + "airway_stage_two/" + fn[:-4] + '.npz')['array']

    blood_one = np.load(top_dict_mask + "blood_vessel_stage_one/" + fn[:-4] + '.npz')['array']
    blood_two = np.load(top_dict_mask + "blood_vessel_stage_two/" + fn[:-4] + '.npz')['array']

    visible_lesions = np.load(top_dict_mask + 'visible_lesion/' + fn[:-4] + '.npz')['array']
    invisible_lesions = np.load(top_dict_mask + 'invisible_lesion/' + fn[:-4] + '.npz')['array']

    invisible_lesions = np.clip(visible_lesions + invisible_lesions, 0, 1)

    enhanced_ct = np.load(top_dict_enhanced + fn[:-4] + '.npz')['array']
    enhanced_ct = np.clip(enhanced_ct, 0, 0.15) / 0.15

    rescaled_ct = np.clip(rescaled_ct + 0.5, 0, 1)

    locations = np.where(invisible_lesions > 0.5)

    z = int(np.median(locations[2]))

    rescaled_ct = rescaled_ct[:, :, z: z + 1]
    airway_one = airway_one[:, :, z: z + 1]
    airway_two = airway_two[:, :, z: z + 1]
    blood_one = blood_one[:, :, z: z + 1]
    blood_two = blood_two[:, :, z: z + 1]
    visible_lesions = visible_lesions[:, :, z: z + 1]
    invisible_lesions = invisible_lesions[:, :, z: z + 1]
    enhanced_ct = enhanced_ct[:, :, z: z + 1]

    #output = highlight.highlight_mask(invisible_lesions, rescaled_ct, 'Y', further_highlight=False)
    #output = highlight.highlight_mask(visible_lesions, output, 'G', further_highlight=True)
    output = highlight.highlight_mask(blood_one, rescaled_ct, 'R', further_highlight=False)
    output = highlight.highlight_mask(airway_one, output, 'B', further_highlight=True)

    #output_2 = highlight.highlight_mask(invisible_lesions, rescaled_ct, 'Y', further_highlight=False)
    #output_2 = highlight.highlight_mask(visible_lesions, output_2, 'G', further_highlight=True)
    output_2 = highlight.highlight_mask(blood_two, rescaled_ct, 'R', further_highlight=False)
    output_2 = highlight.highlight_mask(airway_two, output_2, 'B', further_highlight=True)

    output_3 = highlight.highlight_mask(invisible_lesions, enhanced_ct, 'Y', further_highlight=False)
    output_3 = highlight.highlight_mask(visible_lesions, output_3, 'G', further_highlight=True)

    Functions.image_save(output[:, :, 0, :], high_resolution=True, path=top_dict_save + fn[:-4] + '_1.png')
    Functions.image_save(output_2[:, :, 0, :], high_resolution=True, path=top_dict_save + fn[:-4] + '_2.png')
    Functions.image_save(output_3[:, :, 0, :], high_resolution=True, path=top_dict_save + fn[:-4] + '_3.png')


