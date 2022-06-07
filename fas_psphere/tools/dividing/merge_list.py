# date: 2022-01-13 19:14
# author: liucc
import os
from tqdm import tqdm


def merge_list_in_dir_and_save(list_dir, dst_dir):
    sub_fp_list = []
    for dir_path, dir_names, file_names in os.walk(list_dir, followlinks=True):
        for fn in file_names:
            sub_fp_list.append(os.path.join(dir_path, fn))
    print('found %d lists in ...%s.' % (len(sub_fp_list), dst_dir[-40:]))

    res_dict = dict()
    for sub_fp in sub_fp_list:
        sub_fn = os.path.basename(sub_fp)  # format: det_TYPE_ir_fake_list.txt
        sub_type = sub_fn.split('_')[1]
        # read lines
        with open(sub_fp, 'r') as f:
            sub_list = f.readlines()
        if sub_type not in res_dict:
            res_dict[sub_type] = []
        res_dict[sub_type] += sub_list  # merge

    # save
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    items = tqdm(res_dict.items(), total=len(res_dict))
    for sub_type, sub_list in items:
        items.set_description('saving %s' % sub_type)
        with open(os.path.join(dst_dir, 'det_%s_ir_fake_list.txt' % sub_type), 'w+') as f:
            for sub_fp in sub_list:
                f.write('%s' % sub_fp)
    return


if __name__ == '__main__':
    merge_list_in_dir_and_save(
        list_dir='/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/with_face_info/merge/src',
        dst_dir='/work/liucc/workspace/dataset/face_antiSpoofing/lvmi/fake_images_file_path_list/with_face_info/merge/dst'
    )
























