# date: 2021-12-24 14:21
# author: liucc
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
import multiprocessing

'''
this file is for making data set. input is an image-file-path-list.txt whose element is in format:

image file path, face score(1), face bbox(4), face landmark(10), face occ info(26, lvmi dm5 style)
for instance:
    /a/b/c/d.png, 0.92, 212.2, 212.1, ...
'''

_occ_names_list = ['_A1F1_', '_A1J1_', '_A1I0_', '_A1N1_', '_sung',
                   '_A1K0_', '_A1N2_', '_A1F1E2_', '_A1J1E2_', 'hand', 'paper', 'fist',
                   '_A1I0E2_', '_A1N1E2_', '_A1C1E2_', '_A1K0E2_']


# output: feats_dict is a dict:
#  - key: fp
#  - value: face info[face score(1), face bbox(4), face landmark(10))]
def get_face_info_dict(img_list_txt_fp, ignore_occ_samples: bool):
    print('ignore_occ_samples:', ignore_occ_samples)
    lines = None
    with open(img_list_txt_fp) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    assert lines is not None, img_list_txt_fp

    res_dict = dict()
    for idx, fp_landmark_str in tqdm(enumerate(lines), total=len(lines), desc='loading'):
        try:
            fm_splits = fp_landmark_str.split(',')
            fp = fm_splits[0]
            if ignore_occ_samples:
                area_occ = np.array([float(fm_splits[36]), float(fm_splits[37]), float(fm_splits[38])])
                if (area_occ > 0.6).any() or np.array([e in fp for e in _occ_names_list]).any():
                    continue
            # score(1), bbox(4), landmark(10)
            face_info = []
            for sn in fm_splits[1:16]:
                face_info.append(float(sn))
            res_dict[fp] = face_info
        except Exception as e:
            print(idx, e)
    print('total %d samples, %d valid.' % (len(lines), len(res_dict)))
    return res_dict


def divide_dict(fp_face_info_dict: dict, num_dv: int):
    assert num_dv > 0, num_dv
    elem_size = int(len(fp_face_info_dict) // num_dv)
    dv_list = [elem_size * k for k in range(num_dv)] + [len(fp_face_info_dict)]
    dict_list = [dict() for _ in range(num_dv)]
    keys = list(fp_face_info_dict.keys())
    for n in tqdm(range(num_dv), total=num_dv, desc='dividing dict...'):
        for k in range(dv_list[n], dv_list[n + 1]):
            key = keys[k]
            dict_list[n][key] = fp_face_info_dict[key]
    # delete src
    del fp_face_info_dict
    return dict_list


def expand_bbox(bbox: np.ndarray, ratio, max_w=None, max_h=None):
    box = bbox.copy()
    assert box[0] < box[2] and box[1] < box[3], box
    w, h = ratio * (box[2] - box[0]), ratio * (box[3] - box[1])
    w2, h2 = w / 2, h / 2
    box[0] = max(0.0, box[0] - w2)
    box[1] = max(0.0, box[1] - h2)
    box[2] = min(99999 if max_w is None else max_w, box[2] + w2)
    box[3] = min(99999 if max_h is None else max_h, box[3] + h2)
    return box.astype(bbox.dtype)


def cropping_and_saving_worker(worker_num, fp_face_info_dict, dst_dir, num_dirs_as_id, test_ratio):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    face_info_writer = open(os.path.join(dst_dir, 'face_info_list_%d.txt' % worker_num), 'w+')
    assert face_info_writer is not None
    for idx, (fp, face_info) in enumerate(fp_face_info_dict.items()):
        fp_splits = fp.split('/')
        fid = '-'.join(fp_splits[-num_dirs_as_id:-1])

        rand_v = np.random.random()
        if rand_v < test_ratio:
            t_dst = os.path.join(dst_dir, 'test')
        else:
            t_dst = os.path.join(dst_dir, 'train')

        f_dst = os.path.join(t_dst, fid)
        if not os.path.exists(f_dst):
            os.makedirs(f_dst, exist_ok=False)

        img = cv2.imread(fp)
        if img is None:
            print(' -- [worker %d]: can not read %s.' % (worker_num, fp))
            continue
        dst_fp = os.path.join(f_dst, fp_splits[-1])

        max_h, max_w, _ = img.shape
        ex_bbox = expand_bbox(np.array(face_info[1:5]), max_w=max_w, max_h=max_h, ratio=2.0)
        ex_bbox = ex_bbox.astype(np.int32)
        cropped_img = img[ex_bbox[1]:ex_bbox[3], ex_bbox[0]:ex_bbox[2], :]
        
        face_info = np.array(face_info)
        face_info[1:][0::2] -= ex_bbox[0]
        face_info[1:][1::2] -= ex_bbox[1]

        cv2.imwrite(dst_fp, cropped_img)
        face_info_writer.write('%s, ' % dst_fp)
        for v in face_info:
            face_info_writer.write('%.3f, ' % v)
        face_info_writer.write('\n')
        if idx % 100:
            continue
        print(' -- [worker %d]: %d images processed...' % (worker_num, idx))
    face_info_writer.close()


def make_data_set_to_dir(img_list_txt_fp, dst_dir, num_dirs_as_id=3, test_ratio=-1.0, num_workers=10,
                         ignore_occ_samples=True):
    fp_face_info_dict = get_face_info_dict(img_list_txt_fp, ignore_occ_samples)
    dv_dict_list = divide_dict(fp_face_info_dict, num_workers)
    p_list = []
    for worker_num in range(num_workers):
        src_dict = dv_dict_list[worker_num]
        p = multiprocessing.Process(target=cropping_and_saving_worker,
                                    args=(worker_num, src_dict, dst_dir, num_dirs_as_id, test_ratio))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


def fast_making(argv):
    if len(argv) < 6:
        print('Usage:\n  %s <img_dir> <dst_dir> <num_workers> <test_ratio> <ignore_occ> [num_dirs_as_id=2]' % argv[0])
        print('\n  ref cmd: python xx.py '
              '/a/d/fp_faceInfo_list.txt '
              '/a/d/e/ 15 0.2 1 3')
        exit(1)
    num_dirs_as_id = 2
    if len(argv) > 5:
        num_dirs_as_id = int(argv[6])
    print(argv)
    print('num_dirs_as_id:', num_dirs_as_id)
    make_data_set_to_dir(img_list_txt_fp=argv[1],
                         dst_dir=argv[2],
                         num_workers=int(argv[3]),
                         test_ratio=float(argv[4]),
                         ignore_occ_samples=bool(int(argv[5])),
                         num_dirs_as_id=num_dirs_as_id)
    return


if __name__ == "__main__":
    import sys

    fast_making(sys.argv)
