# date: 2022-01-20 08:48
# author: liucc
import os
import sys
import numpy as np


def update_test_dist_result_by_given(live_dist_fp, fake_dist_fp, given_live_acc: float):
    assert 0 < given_live_acc <= 1.0, given_live_acc
    with open(live_dist_fp, 'r') as f:
        live_lines = f.readlines()
        live_lines = [line.strip() for line in live_lines]
    assert live_lines is not None, live_dist_fp

    with open(fake_dist_fp, 'r') as f:
        fake_lines = f.readlines()
        fake_lines = [line.strip() for line in fake_lines]
    assert fake_lines is not None, fake_dist_fp

    live_dist_list = []
    for line in live_lines:  # line format: 0/1(error/correct), img_fp, dist, radius, margin
        line_split = line.split(',')
        live_dist_list.append(float(line_split[2]))

    live_dist_array = np.sort(np.array(live_dist_list).astype(np.float32))  # ascending order
    # for given_live_acc, threshold should be the given_live_acc*len(live_dist_array)
    thd = live_dist_array[int(given_live_acc * len(live_dist_array))]

    # update
    live_dist_dir, live_dist_fn = os.path.split(live_dist_fp)
    ud_live_dist_fp = os.path.join(live_dist_dir, 'updated_%s' % live_dist_fn)
    fake_dist_dir, fake_dist_fn = os.path.split(fake_dist_fp)
    ud_fake_dist_fp = os.path.join(fake_dist_dir, 'updated_%s' % fake_dist_fn)

    fw_live = open(ud_live_dist_fp, 'w+')
    assert fw_live is not None, ud_live_dist_fp
    fw_fake = open(ud_fake_dist_fp, 'w+')
    assert fw_fake is not None, ud_fake_dist_fp

    for line in live_lines:
        line_split = line.split(',')  # line format: 0/1(error/correct), img_fp, dist, radius, margin
        fw_live.write('%d,%s,%s,%f,%s\n' % (
            int(float(line_split[2]) <= thd), line_split[1], line_split[2], thd, line_split[4]))
    total_cnt, correct_cnt = 1e-6, 0
    for line in fake_lines:
        line_split = line.split(',')  # line format: 0/1(error/correct), img_fp, dist, radius, margin
        is_correct = int(float(line_split[2]) > thd)
        fw_fake.write('%d,%s,%s,%f,%s\n' % (
            is_correct, line_split[1], line_split[2], thd, line_split[4]))
        correct_cnt += is_correct
        total_cnt += 1
    print('for given live acc(%.5f), fake acc is %.5f, thd is %f.' % (
        given_live_acc, float(correct_cnt) / total_cnt, thd))

    fw_fake.close()
    fw_live.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage:\n python %s <test_live_dist_result.txt> <test_fake_dist_result.txt> '
              '<live_acc: float>\n' % sys.argv[0])
        print('NOTE: line format:  0/1(error/correct), img_fp, dist, radius, margin')
        exit(1)
    update_test_dist_result_by_given(
        live_dist_fp=sys.argv[1],
        fake_dist_fp=sys.argv[2],
        given_live_acc=float(sys.argv[3])
    )
















