import numpy as np
import Tool_Functions.Functions as Functions


def up_and_down_merge_png(up_image, down_image):
    # png with shape [height, width, 4], use Function.image_save(image, 'save_path', high_resolution=True)
    shape_up = np.shape(up_image)
    shape_down = np.shape(down_image)

    width = shape_up[1]
    if width < shape_down[1]:
        width = shape_down[1]

    merged_image = np.zeros([shape_up[0] + shape_down[0], width, 4], 'float32')

    merged_image[:shape_up[0], :shape_up[1], :] = up_image
    merged_image[shape_up[0]::, :shape_down[1], :] = down_image

    return merged_image


if __name__ == '__main__':
    image_up = Functions.convert_png_to_np_array('/home/zhoul0a/Desktop/NMI_revision_data/selected_pic/eval_up.png')
    image_down = Functions.convert_png_to_np_array('/home/zhoul0a/Desktop/NMI_revision_data/selected_pic/eval_down.png')
    image_merge = up_and_down_merge_png(image_up, image_down)
    Functions.image_save(image_merge, '/home/zhoul0a/Desktop/NMI_revision_data/selected_pic/merge.png',
                         high_resolution=True)
    exit()
