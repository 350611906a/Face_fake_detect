# date: 2022-01-10 10:24
# author: liucc
import os
import numpy as np


# we have a list of 3D-fake samples, each list is consist of a single type set:
# MRJTT-xxx.txt, MSLMJ-xxx.txt, ...   <- examples
# each type-set is consist of a list of image file paths(here noted by 'instances')

# this project divides them into training and testing set, in testing set there are 15% types does not exist in
# training set, the rest of testing set is consist of types that also exist in training set, but the intersection
# of training and testing set in instance-level is always empty, and instance-level training/testing set ratio is
# around 30%.


def load_type_list_from_dir(types_list_dir):
    type_list = []
    for dir_path, dir_names, file_names in os.walk(types_list_dir, followlinks=True):
        for fn in file_names:
            type_list.append(os.path.join(dir_path, fn))
    assert len(type_list) > 0, types_list_dir
    return type_list


def generate_k_r_percent_indices(length: int, r: float, k: int):
    a = np.array([_ for _ in range(length)]).astype(np.int32)
    np.random.shuffle(a)
    each_len = int(length * r)
    assert k * each_len <= length, '%d > %d, unable to realize k-fold.' % (k * each_len, length)
    print('k = %d, num types no-crossing = %d.' % (k, each_len))
    indices_list = []
    for i in range(0, k):
        indices_list.append(a[i * each_len: (i + 1) * each_len])  # elem: indices
    return indices_list


def get_mean_num_instances(type_list):
    total_num = 0
    for type_fp in type_list:
        with open(type_fp, 'r') as f:
            lines = f.readlines()
        assert lines is not None, type_fp
        total_num += len(lines)
    mean_num = int(total_num / (len(type_list) + 1e-6))
    assert mean_num > 0, type_list
    return mean_num


def pick_up_samples_in_certain_count(type_fp, count: int = 0):
    with open(type_fp, 'r') as f:
        lines = f.readlines()
    assert lines is not None, type_fp
    np.random.shuffle(lines)

    if count < 1:
        count = len(lines)
    count = min(len(lines), count)
    return lines[:count]


def save_str_list_to_file(str_list, dst_fp, need_new_line=False):
    with open(dst_fp, 'w+') as f:
        for sl in str_list:
            if not need_new_line:
                f.write('%s' % sl)
                continue
            f.write('%s\n' % sl)
    return


def save_from_src_fp_to_dst_dir(f_writer, src_fp, dst_dir, keeping_dir_levels: int, need_new_line: bool):
    assert keeping_dir_levels > 0, keeping_dir_levels
    src_fp_split = src_fp.split('/')
    real_dst_dir = os.path.join(dst_dir, ''.join(src_fp[-keeping_dir_levels:-1]))
    if not os.path.exists(real_dst_dir):
        os.makedirs(real_dst_dir)
    dst_fp = os.path.join(real_dst_dir, src_fp_split)
    if need_new_line:
        f_writer.write('%s\n' % dst_fp)
    else:
        f_writer.write('%s' % dst_fp)
    return


def str_list_to_training_and_testing_dir(
        str_list, training_dir, testing_dir, save_fn, training_ratio, need_new_line):
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)
    np.random.shuffle(str_list)
    assert 0 < training_ratio < 1.0, training_ratio
    num_training = int(len(str_list) * training_ratio)
    f_training = open(os.path.join(training_dir, save_fn), 'w+')
    f_testing = open(os.path.join(testing_dir, save_fn), 'w+')
    assert f_training is not None and f_testing is not None, (training_dir, testing_dir, save_fn)
    for idx, s in enumerate(str_list):
        if idx < num_training:
            f = f_training
        else:
            f = f_testing
        if need_new_line:
            f.write('%s\n' % s)
        else:
            f.write('%s' % s)
    f_training.close()
    f_testing.close()
    return


def k_fold(types_list_dir, k=3, dst_dir='./tmp/k_folds'):
    training_dir = os.path.join(dst_dir, 'training')
    testing_dir = os.path.join(dst_dir, 'testing')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)

    type_list = load_type_list_from_dir(types_list_dir=types_list_dir)
    k_indices_list = generate_k_r_percent_indices(len(type_list), r=0.15, k=k)

    # since we have already divided K_folds before, we keep them in indices_list
    old_k_folds = [
        ['MCSSYTM', 'MSLTM', 'MBLGTM', 'MSYSTM'],
        ['MYPMTM', 'MRJMJ', 'MGJMJ', 'MNLTM'],
        ['MBLGMJ', 'B-D', 'M-L', 'A-D'],
    ]
    assert len(old_k_folds) <= len(k_indices_list) and len(old_k_folds[0]) <= len(k_indices_list[0])
    k_indices_list = np.array(k_indices_list).astype(np.int32)
    for i in range(len(old_k_folds)):
        for k in range(len(old_k_folds[i])):
            # looking for old_k_folds' idx in type_list
            idx = 9999
            for j in range(len(type_list)):
                if old_k_folds[i][k] != os.path.basename(type_list[j])[len('det_'):len('det_')+len(old_k_folds[i][k])]:
                    continue
                idx = j
                break
            print(old_k_folds[i][k], os.path.basename(type_list[j]))
            print('[%d, %d] %d -> %d' % (i, k, k_indices_list[i][k], idx))
            k_indices_list[i][k] = idx  # make sure old_k_folds[:] inside

    # num_mean = get_mean_num_instances(type_list)
    num_mean = 0
    # ##################### no type crossing testing set #######################
    for k, indices in enumerate(k_indices_list):
        print('k[%d] type no-crossing set:' % k)
        c_testing_dir = os.path.join(testing_dir, 'k%d' % k)
        if not os.path.exists(c_testing_dir):
            os.makedirs(c_testing_dir)
        c_training_dir = os.path.join(training_dir, 'k%d' % k)
        if not os.path.exists(c_training_dir):
            os.makedirs(c_training_dir)
        for idx in indices:
            type_fn = os.path.basename(type_list[idx])
            print(' -%s' % type_fn)
            samples_list = pick_up_samples_in_certain_count(type_list[idx], count=num_mean)
            # k_samples_list = [k.strip() for k in k_samples_list]
            c_dst_fp = os.path.join(c_testing_dir, type_fn)
            save_str_list_to_file(str_list=samples_list, dst_fp=c_dst_fp, need_new_line=False)

        # #################### type crossing set ########################
        print('k[%d] type crossing set:' % k)
        for j in range(len(type_list)):
            if j in indices:
                continue  # ignore already selected set
            type_fp = type_list[j]
            type_fn = os.path.basename(type_fp)
            print(' -%s' % type_fn)
            samples_list = pick_up_samples_in_certain_count(type_fp, count=num_mean)
            str_list_to_training_and_testing_dir(
                samples_list,
                training_dir=c_training_dir,
                testing_dir=c_testing_dir,
                save_fn=type_fn,
                training_ratio=0.7,
                need_new_line=False
            )
    return


if __name__ == '__main__':
    k_fold(
        types_list_dir='./tmp/divided/',
        k=3,
        dst_dir='./tmp/k_folds'
    )




















