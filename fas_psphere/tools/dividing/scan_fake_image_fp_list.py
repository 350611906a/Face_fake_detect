# date: 2022-01-07 17:53
# author: liucc

import os
import numpy as np
from tqdm import tqdm
from type_list import fake_samples_type_set


# NOTE: following type
# 1. type marked with digit: BCTC-style fake samples
def save_fp_according_to_type_set(fp_list, type_list, dst_dir, desc=''):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # saver
    fw_dict = dict()
    type_list.append('digit')
    for tp in type_list:
        if desc == '':
            fw_fp = os.path.join(dst_dir, '%s_ir_fake_list.txt' % tp)
        else:
            fw_fp = os.path.join(dst_dir, '%s_%s_ir_fake_list.txt' % (desc, tp))
        fw_dict[tp] = open(fw_fp, 'w+')
        if fw_dict[tp] is None:
            print('\n%s failed: \n' % desc, type_list)
            print('\n')
            return
    if desc != '':
        print('saving %s...' % desc)
        bar = enumerate(fp_list)
    else:
        bar = tqdm(enumerate(fp_list), total=len(fp_list), desc='saving')
    for idx, fp in bar:
        try:
            fp_split = fp.split('/')
            type_str = fp_split[-2].split('_')[0]
            # look for type
            fw = None
            for tp in type_list:  # tp example: 'MSGMJ-D'
                jd_len = len(tp[:-2])
                # if tp[:-2] not in type_str:
                if tp[:-2] != type_str[:jd_len]:
                    continue
                fw = fw_dict[tp]
                break
            if fw is None:
                if type_str.isdigit():
                    fw = fw_dict['digit']
                else:
                    continue
            fw.write('%s\n' % fp)
        except Exception as e:
            print(e, fp)
            continue
    for tp in type_list:
        fw_dict[tp].close()
    return


def divide_and_save(raw_fp_list_fp):
    with open(raw_fp_list_fp, 'r') as f:
        fp_list = f.readlines()
        fp_list = [fp.strip() for fp in fp_list]
    assert fp_list is not None, raw_fp_list_fp
    ts_3d, ts_2d = fake_samples_type_set(data_set_dir_root='/storage-server5/dataWarehouse/lvmi/data/fake')
    ts_3d.update(ts_2d)
    type_list = list(ts_3d)
    # sort type_list according to its element length
    type_len_list = [len(t) for t in type_list]
    type_list = np.array(type_list)[np.argsort(type_len_list)[::-1]].tolist()
    save_fp_according_to_type_set(fp_list, type_list, dst_dir='./tmp/divided', desc='')


def scan_fake_images_fp_list(src_dir, dst_dir, dst_fn):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_fp = os.path.join(dst_dir, dst_fn)
    if os.path.exists(dst_fp):
        print('%s already exists, skip scanning.')
        return
    fw = open(dst_fp, 'w+')
    assert fw is not None, dst_fp
    cnt, all_list = 0, []
    print('scanning...')
    for dir_path, dir_names, file_names in os.walk(src_dir, followlinks=True):
        for fn in file_names:
            try:
                if ('_0.png' not in fn) and ('_2.png' not in fn):
                    continue
                src_fp = os.path.join(dir_path, fn)
                # all_list.append(src_fp)
                fw.write('%s\n' % src_fp)
                cnt += 1
                if cnt % 200:
                    continue
                print('\r - %d images found...' % cnt, end='', flush=True)
            except Exception as e:
                print(e)
                continue
    print('\ndone. total %d samples found.' % cnt)
    fw.close()
    return


def fast_scan_and_save():
    dst_dir = './tmp'
    dst_fn = 'raw_ir_fake_fp_list.txt'
    dst_fp = os.path.join(dst_dir, dst_fn)

    if not os.path.exists(dst_fp):
        scan_fake_images_fp_list(
            src_dir='/storage-server5/dataWarehouse/lvmi/data/fake',
            dst_dir=dst_dir,
            dst_fn=dst_fn
        )
    divide_and_save(dst_fp)


if __name__ == '__main__':
    fast_scan_and_save()













