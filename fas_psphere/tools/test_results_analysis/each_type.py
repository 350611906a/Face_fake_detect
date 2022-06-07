# date: 2022-01-19 18:31
# author: liucc
import sys


def get_type(src_sample_fp):
    sample_fp_split = src_sample_fp.split('/')
    sample_raw_type = sample_fp_split[-2]
    e_str = ''
    for e in sample_raw_type:
        if not e.isupper():
            break
        e_str += e
    if len(e_str) == 0:
        return None
    if e_str[0] == "P":
        res = e_str + ("-D" if sample_raw_type[len(e_str)].isdigit() else "-L")
    elif len(e_str) == len(sample_raw_type):
        res = e_str + "-N"
    else:
        res = e_str + ("-D" if sample_raw_type[len(e_str)].isdigit() else "-L")
    return res  # type in str


def print_test_results_in_type(tr_fp):
    with open(tr_fp, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    assert lines is not None, tr_fp

    total_dict, correct_dict = dict(), dict()
    for line in lines:  # line format: 0/1(error/correct), img_fp, dist, radius, margin
        line_split = line.split(',')
        s_type = get_type(line_split[1])
        if s_type not in total_dict:
            total_dict[s_type] = 0
        total_dict[s_type] += 1
        if s_type not in correct_dict:
            correct_dict[s_type] = 0
        correct_dict[s_type] += int(line_split[0])
    # print results
    s_type_list = list(total_dict.keys())
    s_type_list.sort()  # letter ascending order 
    for s_type in s_type_list:
        total_cnt = total_dict[s_type]
        correct_cnt = correct_dict[s_type]
        print('[%15s] acc: %.4f (%d / %d)' % (
            s_type, float(correct_cnt) / (total_cnt + 1e-6), correct_cnt, total_cnt))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:\n python %s <test_dist_result.txt>\n' % sys.argv[0])
        print('NOTE: line format:  0/1(error/correct), img_fp, dist, radius, margin')
        exit(1)
    print_test_results_in_type(sys.argv[1])












