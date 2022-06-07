# date: 2022-01-10 15:14
# author: liucc
import os
import shutil
import multiprocessing


def cp_worker(src_list: list, dst_dir: str, ignore_dir_level: int, desc: str):
    print('%s working...' % desc)
    total = len(src_list)
    for idx, src in enumerate(src_list):
        try:
            src_split = src.split('/')
            c_dst_dir = os.path.join(dst_dir, '/'.join(src_split[ignore_dir_level:len(src_split)-1]))
            if not os.path.exists(c_dst_dir):
                os.makedirs(c_dst_dir)
            c_dst_fp = os.path.join(c_dst_dir, src_split[-1])
            shutil.copyfile(src, c_dst_fp)
            if idx % 1000:
                continue
            print('[%s] %d / %d copied...' % (desc, idx, total))
        except Exception as e:
            print('[%s]:' % desc, e)
            continue
    return


def cp_images_list(src_list_fp: str, dst_dir: str, ignore_dir_level: int, num_workers: int):
    assert ignore_dir_level > 0 and num_workers > 0, (ignore_dir_level, num_workers)
    # loading src_list
    src_fp_list = []
    with open(src_list_fp, 'r') as f:
        src_fp_list = f.readlines()
        src_fp_list = [sp.strip() for sp in src_fp_list]
    # divide to each worker
    d_indices = [0]
    cnt_each_worker = len(src_fp_list) // num_workers
    for idx in range(1, num_workers):
        d_indices.append(idx * cnt_each_worker)
    d_indices.append(len(src_fp_list))

    p_list = []
    for k, di in enumerate(d_indices[1:]):
        desc = 'worker-%d' % k
        print('[%s] processes: [%d : %d]' % (desc, d_indices[k], di))
        c_list = src_fp_list[d_indices[k]:di]
        p = multiprocessing.Process(target=cp_worker,
                                    args=(c_list, dst_dir, ignore_dir_level, desc))
        p_list.append(p)
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


if __name__ == "__main__":
    cp_images_list(
        src_list_fp='./dividing/tmp/raw_ir_fake_fp_list.txt',
        dst_dir='/home/liucc/work/workspace/dataset/face_antiSpoofing/lvmi/fake_20220113',
        ignore_dir_level=3,
        num_workers=15
    )





































