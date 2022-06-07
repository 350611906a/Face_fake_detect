# date: 2022-03-02 17:15
# author: liucc
import os
import sys
import shutil


# this func loads test-list.txt, then copies error-predicted images to dst_dir
def cp_error_images(src_list_fp, dst_dir):
    with open(src_list_fp, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    assert lines is not None, src_list_fp
    print('found %d lines in ...%s.' % (len(lines), src_list_fp[-40:]))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for line in lines:
        # line format: correct{0/1},fp,distance,radius,margin
        spt = line.split(',')

        if int(spt[0]) == 1:
            continue

        src_fp = spt[1]
        src_dir, src_fn = os.path.split(src_fp)
        fn = '%s-%s-%s' % (os.path.basename(src_dir), spt[2], src_fn)
        dst_fp = os.path.join(dst_dir, fn)
        shutil.copyfile(src_fp, dst_fp)
    return


def fast_run(argv):
    if len(argv) < 3:
        print('Usage:\n  python %s <src_list_fp> <dst_dir>' % argv[0])
        print('make sure line format: correct{0/1},fp,distance,radius,margin')
        exit(1)
    cp_error_images(
        src_list_fp=argv[1],
        dst_dir=argv[2]
    )


if __name__ == '__main__':
    fast_run(sys.argv)
















