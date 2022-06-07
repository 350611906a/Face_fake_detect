#coding=utf-8

import os.path as osp
import numpy as np


def save_batch_as_bin(normalized_img_data, img_paths):
    # TODO: refator comment.
    # Save network input data as bin files for verifing model converted 
    # to other format.
    # data shape: [batch_size, channel=1, input_h=256, input_w=256].
    # Saving format [c,h,w].
    for i in range(len(img_paths)):
        assert osp.exists(img_paths[i]), img_paths[i]
        bin_path = img_paths[i].replace('.png', '.bin').replace('.jpg', '.bin')
        data = normalized_img_data[i].detach().numpy()
        # print(data.dtype)
        data.tofile(bin_path)


def read_bin(bin_path):
    # TODO: add comment.
    data = np.fromfile(bin_path, dtype=np.float32)
    data = data.reshape((1, 256, 256))
    return data


def test_read_bin():
    # TODO: move to test directory.
    bin_path = '/ssd1/wangjh/data/ir_fake/lvmi_DM8/lvmi_20220301/IFAS_lvmi_dm8_verify/MBLGMJliuxiaofei_A1C0E1E2_55_inlt_S3/ir_0_20.3_l_0.bin'
    data = read_bin(bin_path)
    print('bin shape: {}'.format(data.shape))
    print('bin type: {}'.format(data.dtype))
    print(data[0][0][0])


if __name__ == '__main__':
    test_read_bin()