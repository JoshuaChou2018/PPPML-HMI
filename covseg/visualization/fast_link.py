import visualization.visualize_3d.highlight_semantics as highlight
import numpy as np


def visualize_lung_tissues(rescaled_array, lung_mask, airway_mask, blood_vessel_mask):

    return_array = highlight.highlight_mask(lung_mask, np.clip(rescaled_array + 0.5, 0, 1), channel='B')
    return_array = highlight.highlight_mask(blood_vessel_mask, return_array, channel='R', further_highlight=True)
    return_array = highlight.highlight_mask(airway_mask, return_array, channel='G', further_highlight=True)

    return return_array  # 4D array like [512, 512 * 2, 512, 3], in range [0, 1]
