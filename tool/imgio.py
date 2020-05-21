import numpy as np
from tensorflow.python.lib.io import file_io
import scipy.misc as misc

def read_img(test_file_path, oss):

    if oss:
        img_obj = file_io.read_file_to_string(test_file_path)
        file_io.write_string_to_file("testimage.jpg", img_obj)
        sci_img = misc.imread("testimage.jpg", mode="RGB")
    else:
        sci_img = misc.imread(test_file_path, mode='RGB')
    x_ = sci_img.shape[0]
    y_ = sci_img.shape[1]
    return x_, y_, np.array(sci_img)

def save_img(img, save_file_path, oss):
    if oss:
        result_obj = file_io.read_file_to_string(save_file_path)
        file_io.write_string_to_file("result.png", result_obj)
        misc.imsave('result.png', img)
        result1_obj = file_io.read_file_to_string('result.png')
        file_io.write_string_to_file(save_file_path, result1_obj)
    else:
        misc.imsave(save_file_path, img)
    return None