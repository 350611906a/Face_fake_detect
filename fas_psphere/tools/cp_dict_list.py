# date: 2022-01-10 16:22
# author: liucc
import os
import shutil
import multiprocessing
import json


def cp_images_dict_worker(src_dict: dict, dst_dir: str, ignore_dir_level: int, desc: str):
    print('[%s] working...' % desc)
    new_dict_saving_dir = os.path.join(dst_dir, 'images_dict')
    if not os.path.exists(new_dict_saving_dir):
        os.makedirs(new_dict_saving_dir, exist_ok=True)
    fw = open(os.path.join(new_dict_saving_dir, '%s.json' % desc), 'w+')
    assert fw is not None, (desc, os.path.join(new_dict_saving_dir, '%s.json' % desc))
    assert ignore_dir_level > 0, ignore_dir_level
    # dict key: id, value: [fp,landmark  fp,landmark  ...]
    total = len(src_dict)
    for c_idx, (str_id, fp_landmark_list) in enumerate(src_dict.items()):
        new_list = []
        for fp_landmark in fp_landmark_list:
            try:
                fp_landmark_split = fp_landmark.split(',')
                src_fp, landmark = fp_landmark_split[0], ','.join(fp_landmark_split[1:])  # /a/b/c,   1,2,3
                src_fp_split = src_fp.split('/')
                c_dst_dir = os.path.join(dst_dir, '/'.join(src_fp_split[ignore_dir_level:len(src_fp_split)-1]))
                if not os.path.exists(c_dst_dir):
                    os.makedirs(c_dst_dir, exist_ok=True)
                c_dst_fp = os.path.join(dst_dir, '/'.join(src_fp_split[ignore_dir_level:]))
                shutil.copyfile(src_fp, c_dst_fp)
                new_list.append('%s,%s' % (c_dst_fp, landmark))
            except Exception as e:
                print('[%s]:' % desc, e)
                continue
        # update dict
        src_dict[str_id] = new_list
        print('[%s]: %d / %d copied...' % (desc, c_idx, total))
    try:
        fw.write(json.dumps(src_dict))
    except Exception as e:
        print('[%s]:' % desc, e)
    fw.close()
    return


def divide_dict(id_fp_lm_dict: dict, num_dv: int):
    assert num_dv > 0, num_dv
    elem_size = int(len(id_fp_lm_dict) // num_dv)
    dv_list = [elem_size * k for k in range(num_dv)] + [len(id_fp_lm_dict)]
    dict_list = [dict() for _ in range(num_dv)]
    keys = list(id_fp_lm_dict.keys())
    for n in range(num_dv):
        for k in range(dv_list[n], dv_list[n+1]):
            key = keys[k]
            dict_list[n][key] = id_fp_lm_dict[key]
    # delete src
    del id_fp_lm_dict
    return dict_list


def cp_images_dict(src_dict_fp: str, dst_dir: str, ignore_dir_level: int, num_workers: int):
    assert ignore_dir_level > 0 and num_workers > 0, (ignore_dir_level, num_workers)
    # loading src_dict
    with open(src_dict_fp, 'r') as f:
        src_dict = json.load(f)
    assert src_dict is not None, src_dict_fp
    dv_dict_list = divide_dict(src_dict, num_workers)

    p_list = []
    for worker_num in range(num_workers):
        src_dict = dv_dict_list[worker_num]
        desc = 'worker_%d' % worker_num
        p = multiprocessing.Process(target=cp_images_dict_worker,
                                    args=(src_dict, dst_dir, ignore_dir_level, desc))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


def fast_cp():
    '''training
    src_dict_fp_list = [
        '/data1/hucs/trainset/lvmi9286_train_all_headshotx_nozhedang.json',
        '/data1/hucs/trainset/lvmi9286_pytorch_headshot_2021_nozhedang.json',
        '/data1/hucs/trainset/lvmi9286_retinaface_headshot_final_all_ori.json',
        '/data1/hucs/trainset/lvmi9286_olds_ori.json',  # oldly
        '/data1/hucs/trainset/lvmi9286_kids_ori.json',  # children
        '/data1/hucs/testset/lvmi9286_kids_ori.json',
        '/data1/hucs/trainset/lvmi_kids/face_info_all.json',
        '/data1/hucs/testset/lvmi_alg-test_kids/face_info_all_refactor.json',
        '/data1/hucs/trainset/lvmi_fxj/face_info_fxj_appendix.json',  # XJ
    ]
    dst_dir_list = [
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/all_headshot_no_occ',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/pytorch_headshot_no_occ',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/retinaface_headshot_no_occ',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/olds_ori',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/kids_ori_trainset',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/kids_ori_testset',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/lvmi_kids',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/lvmi_alg_test_kids',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/lvmi_xj',
    ]
    '''
    '''testing set
    src_dict_fp_list = [
        '/data1/hucs/testset/lvmi9286_retinaface_headshotx_nozhedang_clean.json',
        '/data1/hucs/testset/lvmi9286_old_ori.json',
        '/data1/hucs/testset/lvmi9286_kids_ori_all.json',
        '/data1/hucs/testset/lvmi_xj/face_info_xjc.json',
    ]
    dst_dir_list = [
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/retinaface_headshot_no_occ',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/olds_ori',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/kids_ori_all',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/lvmi_xj',
    ]
    ignore_dir_level = 3
    '''
    '''
    src_dict_fp_list = [
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/all_headshot_no_occ/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/pytorch_headshot_no_occ/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/retinaface_headshot_no_occ/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/olds_ori/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/kids_ori_trainset/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/kids_ori_testset/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/lvmi_kids/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/lvmi_alg_test_kids/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/training_set/lvmi_xj/images_dict/id_fp_landmark.json',
    ]
    dst_dir_list = [
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/all_headshot_no_occ',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/pytorch_headshot_no_occ',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/retinaface_headshot_no_occ',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/olds_ori',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/kids_ori_trainset',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/kids_ori_testset',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/lvmi_kids',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/lvmi_alg_test_kids',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/training_set/lvmi_xj',
    ]
    '''
    src_dict_fp_list = [
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/retinaface_headshot_no_occ/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/olds_ori/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/kids_ori_all/images_dict/id_fp_landmark.json',
        '/storage-server5/dataWhouse/users/liucc/fas/lvmi/live/testing_set/lvmi_xj/images_dict/id_fp_landmark.json',
    ]
    dst_dir_list = [
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/testing_set/retinaface_headshot_no_occ',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/testing_set/olds_ori',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/testing_set/kids_ori_all',
        '/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/live/testing_set/lvmi_xj',
    ]
    ignore_dir_level = 10
    for k in range(len(src_dict_fp_list)):
        cp_images_dict(
            src_dict_fp=src_dict_fp_list[k],
            dst_dir=dst_dir_list[k],
            ignore_dir_level=ignore_dir_level,
            num_workers=15
        )


if __name__ == "__main__":
    fast_cp()





































