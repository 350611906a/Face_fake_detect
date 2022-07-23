# date: 2021-12-27 16:46
# author: liucc

import cv2
import os
import numpy as np
import torch
import random
import torch.utils.data as torch_data
import json
import logging
import sys

# sys.path.append('/ssd1/wangwj/work_dilusense/FAS-trainer/fas_preprocessor/align_and_crop/amba/')
# from ifas_crop_amba_cv28 import align_and_crop_amba_cv28

_verbose = False

def log_debug_info(msg, info):
    # TODO: add check code, only list and 1D ndarray are supported.
    output = [str(a) for a in info]
    output = ','.join(output)
    logging.debug('{}: {}'.format(msg, output))

def landmarks_valid(landmarks):
    # landmarks should be in xy-xy format
    x1, x2, x4, x5 = landmarks[0], landmarks[2], landmarks[6], landmarks[8]
    y1, y2, y4, y5 = landmarks[1], landmarks[3], landmarks[7], landmarks[9]
    if 0 < x2 - x1 and 0 < x5 - x4 and 0 < y4 - y1 and 0 < y5 - y2:
        return True
    return False


def _compute_trans_coef(landmark_pts):
    num_pts = int(len(landmark_pts) / 2)
    assert num_pts == 5
    std_pts = [
        0.290484, 0.263211,
        0.709854, 0.262557,
        0.500565, 0.525255,
        0.317176, 0.751148,
        0.684654, 0.750561
    ]
    a1, a3, a4, c1, c2, c3, c4 = 0., 0., 0., 0., 0., 0., 0.
    for idx in range(num_pts):
        std_x, std_y, x, y = std_pts[idx * 2], std_pts[idx * 2 + 1], landmark_pts[idx * 2], landmark_pts[idx * 2 + 1]
        a1 += std_x * std_x + std_y * std_y
        a3, a4 = a3 + std_x, a4 + std_y
        c1 += x * std_x + y * std_y
        c2 += x * std_y - y * std_x
        c3, c4 = c3 + x, c4 + y
    a1, a3, a4, c1, c2, c3, c4 = a1 / num_pts, a3 / num_pts, a4 / num_pts, c1 / num_pts, c2 / num_pts, c3 / num_pts, c4 / num_pts
    t = a1 - a3 * a3 - a4 * a4
    b1, b2, b3, b4 = 1 / t, a1 / t, -a3 / t, -a4 / t
    a, b = b1 * c1 + b3 * c3 + b4 * c4, b1 * c2 + b4 * c3 - b3 * c4
    dx, dy = b3 * c1 + b4 * c2 + b2 * c3, b4 * c1 - b3 * c2 + b2 * c4
    coef = [a, b, dx, -b, a, dy, 0, 0, 1]
    return coef


# xyxy format
def get_bbox_from_landmark(landmark_pts):
    coef = _compute_trans_coef(landmark_pts)
    center_x = coef[0] * 0.5 + coef[1] * 0.5 + coef[2]
    center_y = coef[3] * 0.5 + coef[4] * 0.5 + coef[5]
    top_left_x, top_left_y = coef[2], coef[5]
    top_right_x, top_right_y = coef[0] + coef[2], coef[3] + coef[5]
    width = np.sqrt((top_left_x - top_right_x) ** 2 + (top_left_y - top_right_y) ** 2)
    xyxy = [int(center_x - width / 2 + 0.5), int(center_y - width / 2 + 0.5),
            int(center_x + width / 2 + 0.5), int(center_y + width / 2 + 0.5)]
    return xyxy


# pt1 and pt2 forms line1, make a vertical line2 from nose pt to line1, this fun return cross point
def nose_between_2_pts(pt1, pt2, nose_pt):
    x1, y1 = pt1
    x2, y2 = pt2
    nx, ny = nose_pt
    if abs(x2 - x1) < 2:
        return (x1 + x2) // 2, ny
    if abs(y2 - y1) < 2:
        return nx, (y1 + y2) // 2
    k = (y2 - y1) / (x2 - x1)
    x0 = (ny - y1 + k * x1 + nx / k) / (k + 1 / k)
    y0 = y1 + k * (x0 - x1)
    return x0, y0


def cal_yaw_by_landmark(landmark):
    left_eye, right_eye, nose = landmark[:2], landmark[2:4], landmark[4:6]
    nose_between_eyes = nose_between_2_pts(left_eye, right_eye, nose)
    d1 = np.sqrt((left_eye[0] - nose_between_eyes[0]) ** 2 + (left_eye[1] - nose_between_eyes[1]) ** 2)
    d2 = np.sqrt((right_eye[0] - nose_between_eyes[0]) ** 2 + (right_eye[1] - nose_between_eyes[1]) ** 2)
    d0 = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
    if d1 >= d0:
        return np.pi / 2
    if d2 >= d0:
        return -np.pi / 2
    # return np.tanh((d1 - d2) / (d0 + 1e-6)) * (np.pi / 2)
    return (((d1 - d2) / (d0 + 1e-6)) * (np.pi / 2)) / np.pi * 180


def cal_pitch_by_landmark(landmark):
    lex, ley, rex, rey, nose_x, nose_y, lmx, lmy, rmx, rmy = landmark
    nbm = nose_between_2_pts((lmx, lmy), (rmx, rmy), (nose_x, nose_y))  # nbm -> nose between mouth
    nbe = nose_between_2_pts((lex, ley), (rex, rey), (nose_x, nose_y))  # nbe -> nose between eyes
    d1 = np.sqrt((nose_x - nbm[0]) ** 2 + (nose_y - nbm[1]) ** 2)
    d2 = np.sqrt((nose_x - nbe[0]) ** 2 + (nose_y - nbe[1]) ** 2)
    d0 = np.sqrt((nbm[0] - nbe[0]) ** 2 + (nbm[1] - nbe[1]) ** 2)
    if d1 >= d0:
        return np.pi / 2
    if d2 >= d0:
        return -np.pi / 2
    # return np.tanh((d1 - d2) / (d0 + 1e-6)) * (np.pi / 2)
    return (((d1 - d2) / (d0 + 1e-6)) * (np.pi / 2)) / np.pi * 180


def random_rotate_image(img, bbox, landmark, max_angle):
    # if angle_choices is None:
    #     angle_choices = [-45, -35, -25, -15, 15, 25, 35, 45]
    assert bbox.shape[1] == 4 and landmark.shape[1] == 10
    # bbox/landmark shape: [num faces, 4] / [num faces, 10]
    # ra = np.random.choice(angle_choices)
    ra = (random.random() - 0.5) * 2 * max_angle  # [-max_angle, max_angle]
    center_tuple = (img.shape[1] // 2, img.shape[0] // 2)
    # rotate bbox and landmark
    ra_radian = ra / 180 * np.pi
    rmx = np.array([[np.cos(ra_radian), -np.sin(ra_radian)],
                    [np.sin(ra_radian), np.cos(ra_radian)]]).astype(np.float32)
    num_faces = bbox.shape[0]
    center = np.array(center_tuple).astype(np.float32)
    give_up = False
    bbox_copy, landmark_copy = bbox.copy(), landmark.copy()
    for n in range(num_faces):
        # rotate_matrix is based on (0, 0), hence coordinates' offset is concerned
        for i in range(landmark.shape[1] // 2):
            landmark[n, i * 2:i * 2 + 2] = (landmark[n, i * 2] - center[0]) * rmx[0] + \
                                           (landmark[n, i * 2 + 1] - center[1]) * rmx[1] + center
        if (landmark[n] < 0).any() or (landmark[n, 0::2] > img.shape[1]).any() or (
                landmark[n, 1::2] > img.shape[0]).any():
            give_up = True
            break
        bbox[n] = get_bbox_from_landmark(landmark[n])
        eye_distance = np.sqrt((landmark[n, 0] - landmark[n, 2]) ** 2 + (landmark[n, 1] - landmark[n, 3]) ** 2)
        if bbox[n, 2] - bbox[n, 0] < eye_distance or bbox[n, 3] - bbox[n, 1] < eye_distance:
            give_up = True
            break
    if give_up:
        return img, bbox_copy, landmark_copy
    m = cv2.getRotationMatrix2D(center_tuple, ra, scale=1)
    wa_img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    return wa_img, bbox, landmark


def get_similarity_transform(src_pts, dst_pts):
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    assert src_pts.ndim == 2
    assert dst_pts.ndim == 2
    assert src_pts.shape[-1] == 2
    assert dst_pts.shape[-1] == 2

    npts = src_pts.shape[0]
    A = np.empty((npts * 2, 4))
    b = np.empty((npts * 2,))
    for k in range(npts):
        A[2 * k + 0] = [src_pts[k, 0], -src_pts[k, 1], 1, 0]
        A[2 * k + 1] = [src_pts[k, 1], src_pts[k, 0], 0, 1]
        b[2 * k + 0] = dst_pts[k, 0]
        b[2 * k + 1] = dst_pts[k, 1]

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    xform_coeff = np.empty((3, 3))
    xform_coeff[0] = [x[0], -x[1], x[2]]
    xform_coeff[1] = [x[1], x[0], x[3]]
    xform_coeff[2] = [0, 0, 1]
    return xform_coeff


def align_and_crop_normal(image, landmarks, std_landmarks, align_size):
    landmarks = np.asarray(landmarks)
    std_landmarks = np.asarray(std_landmarks)
    xform_coeff = get_similarity_transform(landmarks, std_landmarks)
    image_cropped = cv2.warpAffine(image, xform_coeff[:2, :], dsize=align_size)
    return image_cropped, xform_coeff


def align_and_crop(image, landmarks, align_size=224):
    # std_landmarks for 112x112 shape
    std_landmarks = np.array([[8 + 30.2946, 51.6963],  # left eye
                              [8 + 65.5318, 51.5014],  # right eye
                              [8 + 48.0252, 71.7366],  # nose tip
                              [8 + 33.5493, 92.3655],  # left mouth corner
                              [8 + 62.7299, 92.2041]], dtype=np.float32)  # right mouth corner
    std_landmarks *= 0.5  # this make we have wider border
    std_landmarks += 112 * 0.25  # upper scale is based on origin point, shift to center

    std_landmarks *= align_size / 112
    landmarks = np.asarray(landmarks).reshape(5, 2)
    image_cropped, xform_coeff = align_and_crop_normal(image, landmarks, std_landmarks, (align_size, align_size))
    # transform landmark
    landmark_src = np.array(landmarks).astype(np.float32).reshape((5, 2))
    landmark = []
    for ls in landmark_src:
        landmark.append((*ls, 1))
    landmark_src = np.array(landmark).astype(np.float32).reshape((5, 3)).T
    landmark_src = np.dot(xform_coeff, landmark_src).T
    landmark = []
    for ls in landmark_src:
        landmark.append(ls[:2])
    landmark = np.array(landmark).reshape((-1)).astype(np.int32)
    return image_cropped, landmark


# see xx_tools/cvGenRandomArea.py for fast test
def random_mask(width, height, once_change, surely_thin=False):
    raw_img = np.zeros((height, width, 1), dtype=np.float32)
    raw_img_t = np.zeros((width, height, 1), dtype=np.float32)

    num_sub_w = width  # // 4
    num_sub_h = height  # // 4

    def _do_mask(img):
        assert num_sub_w > 0 and num_sub_h > 0, "%d %d" % (num_sub_w, num_sub_h)
        h, w, c = img.shape
        elem_w = w // num_sub_w
        elem_h = h // num_sub_h

        rand_sx_s = 0 if not surely_thin else max(elem_w - 1, 0)
        rand_sx_e = elem_w
        rand_ex_s = (num_sub_w - 1) * elem_w
        rand_ex_e = w - 1 if not surely_thin else min(rand_ex_s, w - 1)
        for idx in range(num_sub_h):
            sx, ex = random.randint(rand_sx_s, rand_sx_e), random.randint(rand_ex_s, rand_ex_e)
            if ex - sx < 6:
                img[idx * elem_h: (idx + 1) * elem_h, sx:ex, :] = 1
            else:
                img[idx * elem_h: (idx + 1) * elem_h, sx, :] = 0.1
                img[idx * elem_h: (idx + 1) * elem_h, sx + 1, :] = 0.4
                img[idx * elem_h: (idx + 1) * elem_h, sx + 2, :] = 0.9
                img[idx * elem_h: (idx + 1) * elem_h, ex - 1, :] = 0.1
                img[idx * elem_h: (idx + 1) * elem_h, ex - 2, :] = 0.4
                img[idx * elem_h: (idx + 1) * elem_h, ex - 3, :] = 0.9
                img[idx * elem_h: (idx + 1) * elem_h, sx + 3:ex - 3, :] = 1
            # img[idx * elem_h: (idx + 1) * elem_h, sx:ex, :] = 1

            rand_sx_s = random.randint(int(max(sx - once_change, 0)), sx)
            rand_sx_e = random.randint(sx, int(min(sx + once_change, w - 1)))
            rand_ex_s = random.randint(int(max(ex - once_change, 0)), ex)
            rand_ex_e = random.randint(ex, int(min(ex + once_change, w - 1)))
        return img

    mask_img = _do_mask(raw_img)
    tmp = num_sub_h
    num_sub_h = num_sub_w
    num_sub_w = tmp
    mask_img_t = _do_mask(raw_img_t)
    mask_img_t_t = mask_img_t.transpose((1, 0, 2))
    pos_mask = (mask_img > 0) & (mask_img_t_t > 0)
    mask = mask_img + mask_img_t_t
    mask /= 2
    mask[~pos_mask] = 0

    return mask


def random_occ_img(src_img, occ_ctx_img, occ_center, occ_w, occ_h, only_norm=False):

    occ_center = np.array(occ_center).astype(np.int32)
    if occ_w < 10 or occ_h < 10 or (occ_center < 0).any():
        return src_img

    f_mask = random_mask(occ_w, occ_h, once_change=1)
    occ_ctx_img = cv2.resize(occ_ctx_img.copy(), (f_mask.shape[1], f_mask.shape[0]))
    if f_mask.shape[0] < 10 or f_mask.shape[1] < 10:
        return src_img

    # make sure occ_ctx_img is within src_img
    if src_img.shape[0] < occ_ctx_img.shape[0] or src_img.shape[1] < occ_ctx_img.shape[1]:
        return src_img
    occ_center[0] = max(int(np.ceil(occ_ctx_img.shape[1] / 2)), occ_center[0])
    occ_center[0] = min(src_img.shape[1] - int(np.ceil(occ_ctx_img.shape[1] / 2)), occ_center[0])
    occ_center[1] = max(int(np.ceil(occ_ctx_img.shape[0] / 2)), occ_center[1])
    occ_center[1] = min(src_img.shape[0] - int(np.ceil(occ_ctx_img.shape[0] / 2)), occ_center[1])

    mask = (f_mask * 255).astype(np.uint8)
    if only_norm:
        clone_method = cv2.NORMAL_CLONE

    else:
        clone_method = cv2.NORMAL_CLONE if np.random.random() < 0.7 else cv2.MIXED_CLONE
    try:
        t = cv2.seamlessClone(occ_ctx_img, src_img, mask, tuple(occ_center), clone_method)
        # cv2.imwrite("seamlessClone.png", t)

    except Exception as e:
        if np.random.random() < 0.1:
            print('random occ img:', e)
        return src_img
    
    return t


# list_fp line format: img_fp, score, bbox[4], landmark[10]
def get_fp_landmark_tuples(list_fp, max_samples_each_list, verbose):
    with open(list_fp, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    assert lines is not None, list_fp

    fp_landmark_tuples = list()
    if verbose:
        print('loading data fp-landmark from ...%s....' % list_fp[-400:])
    cnt = 0
    np.random.shuffle(lines)
    for idx, line in enumerate(lines):
        if ';' in line:
            parts = line.split(';')
            fp = parts[0]
            landmark = parts[2].split(',')
            landmark = [float(f) for f in landmark]
        else:
            lst = line.split(',')
            fp, landmark = lst[0], []
            for str_lm in lst[6:16]:
                landmark.append(float(str_lm))
				
        if len(landmark) != 10:
            continue
        if not landmarks_valid(landmark):
            continue
        fp_landmark_tuples.append((fp, landmark))
        cnt += 1
        if cnt > max_samples_each_list:
            break
        if idx % 200:
            continue
        if verbose:
            print('\r - %d samples found.' % idx, end='', flush=True)
    if verbose:
        print('\ndone, %d samples found.' % len(fp_landmark_tuples))
    return fp_landmark_tuples

def get_fp_landmark_tuples_wjh(list_fp, max_samples_each_list, verbose):
    with open(list_fp, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    assert lines is not None, list_fp

    fp_landmark_tuples = list()
    if verbose:
        print('loading data fp-landmark from ...%s....' % list_fp[-40:])
    cnt = 0
    np.random.shuffle(lines)
    for idx, line in enumerate(lines):
        # lst = line.split(',')
        lst = line.split(' ')[0].split(';')
        fp, landmark = lst[0], []
        # for str_lm in lst[6:16]:
        for str_lm in lst[2].split(','):
            landmark.append(float(str_lm))
        if len(landmark) != 10:
            continue
        if not landmarks_valid(landmark):
            continue
        fp_landmark_tuples.append((fp, landmark))
        cnt += 1
        if cnt > max_samples_each_list:
            break
        if idx % 200:
            continue
        if verbose:
            print('\r - %d samples found.' % idx, end='', flush=True)
    if verbose:
        print('\ndone, %d samples found.' % len(fp_landmark_tuples))
    return fp_landmark_tuples

def get_and_align_fp_landmark_tuples_from_fp_list_test(list_fp_list, max_samples_each_list, need_align=True, verbose=True):
    # list_fp_list is consist of file_paths of samples list
    # with open(list_fp_list, 'r') as f:
    #     list_fp_lines = f.readlines()
    #     list_fp_lines = [line.strip() for line in list_fp_lines]
    # assert list_fp_lines is not None, list_fp_list
    list_fp_lines = [list_fp_list]
    tuples_list, max_cnt = [], 0
    for list_fp in list_fp_lines:
        tl = get_fp_landmark_tuples_wjh(list_fp, max_samples_each_list=max_samples_each_list, verbose=verbose)
        max_cnt = max(max_cnt, len(tl))
        tuples_list.append(tl)
    # align to the same len
    cat_list = []
    for tp_list in tuples_list:
        if need_align:
            tp_list = tp_list * (max_cnt // len(tp_list))
            tp_list = tp_list + tp_list[:max_cnt - len(tp_list)]
        cat_list += tp_list
    return cat_list


def get_and_align_fp_landmark_tuples_from_fp_list(list_fp_list, max_samples_each_list, need_align=True, verbose=True):
    # list_fp_list is consist of file_paths of samples list
    with open(list_fp_list, 'r') as f:
        list_fp_lines = f.readlines()
        list_fp_lines = [line.strip() for line in list_fp_lines]
    assert list_fp_lines is not None, list_fp_list
    tuples_list, max_cnt = [], 0
    for list_fp in list_fp_lines:
        tl = get_fp_landmark_tuples(list_fp, max_samples_each_list=max_samples_each_list, verbose=verbose)
        max_cnt = max(max_cnt, len(tl))
        tuples_list.append(tl)
    # align to the same len
    cat_list = []
    for tp_list in tuples_list:
        if need_align:
            tp_list = tp_list * (max_cnt // len(tp_list))
            tp_list = tp_list + tp_list[:max_cnt - len(tp_list)]
        cat_list += tp_list
    return cat_list

def get_fp_landmark_tuples_from_json_list(json_fp_list_fp, max_samples_each_id: int, verbose=True):
    assert max_samples_each_id > 0, max_samples_each_id
    # each json format:
    # key: id, value is a list of this id image instance and its landmark: ['fp,landmark', ...]
    with open(json_fp_list_fp, 'r') as f:
        json_fp_list = f.readlines()
        json_fp_list = [j.strip() for j in json_fp_list]
    assert json_fp_list is not None, json_fp_list_fp

    fp_landmark_tuples = list()
    for json_fp in json_fp_list:     # json_fp —— 判断的依据
        f = open(json_fp, 'r')
        if f is None:
            print('can not load json from', json_fp)
            continue
        id_fp_landmark_dict = json.load(f)
        f.close()
        if verbose:
            print('loading from ...%s..., total %d ids' % (json_fp[-400:], len(id_fp_landmark_dict)))
        num_ignore_big_yaw, num_ignore_big_pitch = 0, 0
        for c_id, fp_landmark_str_list in id_fp_landmark_dict.items():
            cnt = 0
            np.random.shuffle(fp_landmark_str_list)
            for fp_landmark_str in fp_landmark_str_list:
                if ';' in fp_landmark_str:
                    parts = fp_landmark_str.split(';')
                    fp = parts[0]
                    landmark = [float(f) for f in parts[2].split(',')]
                else:
                    if 'face_info-w_occ-w_pose_right_path' in json_fp:
                        fp_landmark_split = fp_landmark_str.split(',')
                        fp = fp_landmark_split[0]
                        landmark = [float(e) for e in fp_landmark_split[2:12]]
                    else:
                        fp_landmark_split = fp_landmark_str.split(',')
                        fp = fp_landmark_split[0]
                        landmark = [float(e) for e in fp_landmark_split[1:11]]                        

                file_name = os.path.basename(fp)
                if '_out_' in file_name or '_A1I0' in file_name:
                    continue
                assert len(landmark) == 10, fp_landmark_split

                # ignore samples with too large angle
                if abs(cal_yaw_by_landmark(landmark)) > 50:
                    num_ignore_big_yaw += 1
                    continue
                if abs(cal_pitch_by_landmark(landmark)) > 60:
                    num_ignore_big_pitch += 1
                    continue

                fp_landmark_tuples.append((fp, landmark))
                cnt += 1

                if 'face_info-w_occ-w_pose_right_path' in json_fp:
                    if cnt < 50:    # 表示在冲冲的标准人脸数据库中，json的每个id仅仅只取20个
                        continue
                    else:
                        break

                if cnt < max_samples_each_id:
                    continue
                break
        if verbose:
            print('done. ignore: yaw(%d), pitch(%d), current samples: %d.' % (
                num_ignore_big_yaw, num_ignore_big_pitch, len(fp_landmark_tuples)))
    return fp_landmark_tuples


def expand_bbox(bbox: np.ndarray, ratio: float, max_w=None, max_h=None):
    assert ratio >= 1.0, ratio
    box = bbox.copy()
    assert box[0] < box[2] and box[1] < box[3], box
    w, h = (ratio-1.0) * (box[2] - box[0]), (ratio-1.0) * (box[3] - box[1])
    w2, h2 = w / 2, h / 2
    box[0] = max(0.0, box[0] - w2)
    box[1] = max(0.0, box[1] - h2)
    box[2] = min(99999 if max_w is None else max_w, box[2] + w2)
    box[3] = min(99999 if max_h is None else max_h, box[3] + h2)
    return box.astype(bbox.dtype)


def center_crop(src_img, center, cropping_dim, dst_dim):
    assert len(src_img.shape) == 3, src_img.shape
    cropping_dim = min(cropping_dim, dst_dim)
    dst = np.zeros((dst_dim, dst_dim, src_img.shape[2]), dtype=np.float32) + 127.5

    sx = max(center[0] - cropping_dim // 2, 0)
    sy = max(center[1] - cropping_dim // 2, 0)
    cropped_img = src_img.copy()[sy:sy + cropping_dim, sx:sx + cropping_dim, :]

    real_h, real_w = cropped_img.shape[:2]
    sx, sy = (dst_dim - real_w) // 2, (dst_dim - real_h) // 2
    if sx < 0 or sy < 0:
        dst = cv2.resize(cropping_dim, (dst_dim, dst_dim))  # should be seldom happened
    else:
        dst[sy:sy + real_h, sx:sx + real_w, :] = cropped_img
    return dst


def random_crop(src_img, landmark, dst_dim):
    if landmark[0] == 0:
        return src_img, landmark

    landmark = np.array(landmark).astype(np.float32)
    assert len(src_img.shape) == 3, src_img.shape
    sh, sw, _ = src_img.shape
    # expand
    src_img = src_img.copy()
    if sh < dst_dim or sw < dst_dim:
        landmark[0::2] *= float(dst_dim / sw)
        landmark[1::2] *= float(dst_dim / sh)
        src_img = cv2.resize(src_img, (dst_dim, dst_dim))
        sh, sw, _ = src_img.shape
    # random crop
    sx = random.randint(0, sw - dst_dim)
    sy = random.randint(0, sh - dst_dim)
    dst_img = src_img[sy:sy + dst_dim, sx:sx + dst_dim, :]
    # landmark[0::2] -= sx  # landmark could be random
    # landmark[1::2] -= sy
    landmark[0::2] *= float(dst_dim / sw)
    landmark[1::2] *= float(dst_dim / sh)
    return dst_img, landmark

def random_crop_new(src_img, landmark, dst_dim):
    if landmark[0] == 0:
        return src_img, landmark

    landmark = np.array(landmark).astype(np.float32)
    assert len(src_img.shape) == 3, src_img.shape
    sh, sw, _ = src_img.shape
    # expand
    src_img = src_img.copy()
    if sh < dst_dim or sw < dst_dim:
        landmark[0::2] *= float(dst_dim / sw)
        landmark[1::2] *= float(dst_dim / sh)
        src_img = cv2.resize(src_img, (dst_dim, dst_dim))
        sh, sw, _ = src_img.shape
        # random crop
        sx = random.randint(0, sw - dst_dim)
        sy = random.randint(0, sh - dst_dim)
        dst_img = src_img[sy:sy + dst_dim, sx:sx + dst_dim, :]
        landmark[0::2] *= float(dst_dim / sw)
        landmark[1::2] *= float(dst_dim / sh)
    else:
        # random crop
        sx = random.randint(0, sw - dst_dim)
        sy = random.randint(0, sh - dst_dim)
        dst_img = src_img[sy:sy + dst_dim, sx:sx + dst_dim, :]
        landmark[0::2] -= sx
        landmark[1::2] -= sy
    return dst_img, landmark


def brightness_distortion(src_img, alpha=1.0, beta=0.0):
    img = src_img.copy().astype(np.float32) * alpha + beta
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(src_img.dtype)


def random_bright_or_dark_landmark_area(src_img, landmark):
    if landmark[0] == 0:
        return src_img

    landmark = np.array(landmark).astype(np.int32)
    src_img = src_img.copy()
    d_eyes = np.sqrt((landmark[2]-landmark[0]) ** 2 + (landmark[3]-landmark[1]) ** 2)
    d_eye_mouth = np.sqrt((landmark[6]-landmark[0]) ** 2 + (landmark[7]-landmark[1]) ** 2)
    area_w = max(int(d_eyes), int(d_eye_mouth)) // 5 * 4
    rd_mask = random_mask(area_w, area_w, once_change=1)
    rd_mask *= random.uniform(0.4, 0.9)
    # possible centers: left/right eye, nose tip, mouth center
    pcs = [*landmark[:6], (landmark[6]+landmark[8])//2, (landmark[7]+landmark[9])//2]
    center = np.array(pcs).reshape(4, 2)[random.randint(0, 3)]  # randint(0, 3) = random {0,1,2,3}
    sx, sy = max(0, center[0] - area_w // 2), max(0, center[1] - area_w // 2)
    ex = min(src_img.shape[1], center[0] + area_w // 2)
    ey = min(src_img.shape[0], center[1] + area_w // 2)
    
    if (ex - sx < 2) or (ey - sy < 2):
        print('random_bright_or_dark_landmark_area: error size.')
        return src_img

    if np.random.random() < 0.5:
        mask_img = src_img.astype(np.float32)[sy:ey, sx:ex, :] * cv2.resize(rd_mask, (ex - sx, ey - sy))[..., None]
        mask_img[mask_img > 255] = 255
        mask_img[mask_img < 0] = 0
        src_img[sy:ey, sx:ex] -= mask_img.astype(np.uint8)
    else:
        rd_mask += 1
        mask_img = src_img.astype(np.float32)[sy:ey, sx:ex, :] * cv2.resize(rd_mask, (ex - sx, ey - sy))[..., None]
        mask_img[mask_img > 255] = 255
        mask_img[mask_img < 0] = 0
        src_img[sy:ey, sx:ex] = mask_img.astype(np.uint8)
    return src_img

def random_bright_left_or_right_glasses_landmark_area(src_img, landmark):
    if landmark[0] == 0:
        return src_img

    landmark = np.array(landmark).astype(np.int32)
    src_img = src_img.copy()
    d_eyes = np.sqrt((landmark[2]-landmark[0]) ** 2 + (landmark[3]-landmark[1]) ** 2)
    point_size = int(d_eyes // 16)
    point_color = (255, 255, 255)
    thickness = 12
    
    if np.random.random() < 0.5:
        center = (int(landmark[0])-5, int(landmark[1])+int(d_eyes // 6))
    else:
        center = (int(landmark[2])+5, int(landmark[1])+int(d_eyes // 6))

    # cv2.imwrite("random_glasses_before.png", src_img)  
    cv2.circle(src_img, center, point_size, point_color, thickness)
    # cv2.imwrite("random_glasses_after.png", src_img)
    
    return src_img


def get_occ_live_img_as_fake(live_img, landmark, non_live_img):
    if landmark[0] == 0:
        return live_img

    landmark = np.array(landmark).astype(np.int32)
    live_img, non_live_img = live_img.copy(), non_live_img.copy()
    d_eyes = np.sqrt((landmark[8]-landmark[0]) ** 2 + (landmark[9]-landmark[1]) ** 2)
    d_eye_mouth = np.sqrt((landmark[6]-landmark[0]) ** 2 + (landmark[7]-landmark[1]) ** 2)
    area_w = max(int(d_eyes), int(d_eye_mouth))
    assert non_live_img.shape[0] >= area_w and non_live_img.shape[1] >= area_w, (non_live_img.shape, area_w)

    # get occ content img from non_live_img
    sx = random.randint(0, non_live_img.shape[1]-area_w)
    sy = random.randint(0, non_live_img.shape[0]-area_w)

    occ_img = non_live_img[sy:sy+area_w, sx:sx+area_w, :]
    # random occ landmark area 
    center = (random.randint(landmark[0], max(landmark[8], landmark[0])), 
              random.randint(landmark[1], max(landmark[9], landmark[1])))
    occ_fake_img = random_occ_img(live_img, occ_img, center, area_w, area_w, only_norm=True)

    return occ_fake_img


'''
def prepare_img_stack(src_img, landmark, img_dim, need_augment=True, non_face_img=None):
    src_img, landmark = align_and_crop(src_img, landmark, align_size=512)
    bbox_from_lm = get_bbox_from_landmark(landmark)

    if need_augment:
        if np.random.random() < 0.3:
            img_en = cv2.imencode('.jpg', src_img.copy())
            if not img_en[0]:
                return None
            src_img = cv2.imdecode(img_en[1], 1)
        if False and (non_face_img is not None) and (np.random.random() < 0.1):
            src_img = random_occ_img(
                src_img.copy(), non_face_img,
                (int(random.uniform(bbox_from_lm[0], bbox_from_lm[2])),
                 int(random.uniform(bbox_from_lm[1], bbox_from_lm[2]))),
                int((bbox_from_lm[2] - bbox_from_lm[0]) * 0.5),
                int((bbox_from_lm[3] - bbox_from_lm[1]) * 0.5))
    bbox = expand_bbox(np.array(bbox_from_lm).astype(np.int32), ratio=1.4,
                       max_w=src_img.shape[1], max_h=src_img.shape[0])
    g_img = src_img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    if g_img is None or 0 in g_img.shape:
        return None
    if g_img.shape[0] != img_dim or g_img.shape[1] != img_dim:
        g_img = cv2.resize(g_img, (img_dim, img_dim))

    img_stack = g_img[..., :1]

    if need_augment:
        if np.random.random() < 0.5:
            img_stack = img_stack[:, ::-1, :]
        if np.random.random() < 0.5:
            img_stack = img_stack[::-1, :, :]
        if non_face_img is not None:  # this indicates current is not live samples, we can do any augments we like
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, alpha=random.uniform(0.5, 1.5))
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, beta=random.uniform(-32, 32))
            if np.random.random() < 0.4:  # slightly change pixels
                origin_type = img_stack.dtype
                img_stack = img_stack.astype(np.float32) + np.random.rand(*img_stack.shape) * 20 - 10
                img_stack[img_stack < 0] = 0
                img_stack[img_stack > 255] = 255
                img_stack = img_stack.astype(origin_type)

    img_stack = (img_stack.astype(np.uint8).astype(np.float32) - 127.5) / 127.5
    img_stack = img_stack.transpose((2, 0, 1))
    return img_stack  # shape: [1, img_dim, img_dim]
'''


def prepare_img_stack(src_img, landmark, img_dim, need_augment=True, non_face_img=None):
    if landmark[0] == 0:
        if src_img.shape[0] != img_dim or src_img.shape[1] != img_dim:
            g_img = cv2.resize(src_img, (img_dim, img_dim))
            img_stack = g_img[..., :1]
        else:
            img_stack = src_img[..., :1]        
        img_stack = (img_stack.astype(np.uint8).astype(np.float32) - 127.5) / 127.5
        img_stack = img_stack.transpose((2, 0, 1))
        return img_stack  # shape: [1, img_dim, img_dim]

    # crop 1.4 * box    
    # align_and_crop     
    # resize 256*256   

    src_img, landmark = align_and_crop(src_img, landmark, align_size=512)
    bbox_from_lm = get_bbox_from_landmark(landmark)

    if need_augment:
        if np.random.random() < 0.3:
            img_en = cv2.imencode('.jpg', src_img.copy())
            if not img_en[0]:
                return None
            src_img = cv2.imdecode(img_en[1], 1)
        if False and (non_face_img is not None) and (np.random.random() < 0.1):
            src_img = random_occ_img(
                src_img.copy(), non_face_img,
                (int(random.uniform(bbox_from_lm[0], bbox_from_lm[2])),
                 int(random.uniform(bbox_from_lm[1], bbox_from_lm[2]))),
                int((bbox_from_lm[2] - bbox_from_lm[0]) * 0.5),
                int((bbox_from_lm[3] - bbox_from_lm[1]) * 0.5))
    bbox = expand_bbox(np.array(bbox_from_lm).astype(np.int32), ratio=1.4,
                       max_w=src_img.shape[1], max_h=src_img.shape[0])
    g_img = src_img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    if g_img is None or 0 in g_img.shape:
        return None
    if g_img.shape[0] != img_dim or g_img.shape[1] != img_dim:
        g_img = cv2.resize(g_img, (img_dim, img_dim))

    img_stack = g_img[..., :1]

    if need_augment:
        if np.random.random() < 0.5:
            img_stack = img_stack[:, ::-1, :]
        if np.random.random() < 0.5:
            img_stack = img_stack[::-1, :, :]
        if non_face_img is not None:  # this indicates current is not live samples, we can do any augments we like
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, alpha=random.uniform(0.5, 1.5))
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, beta=random.uniform(-32, 32))
            if np.random.random() < 0.4:  # slightly change pixels
                origin_type = img_stack.dtype
                img_stack = img_stack.astype(np.float32) + np.random.rand(*img_stack.shape) * 20 - 10
                img_stack[img_stack < 0] = 0
                img_stack[img_stack > 255] = 255
                img_stack = img_stack.astype(origin_type)

    img_stack = (img_stack.astype(np.uint8).astype(np.float32) - 127.5) / 127.5
    img_stack = img_stack.transpose((2, 0, 1))
    return img_stack  # shape: [1, img_dim, img_dim]

def align_and_crop_test(image, landmarks, align_size=224):
    # std_landmarks for 112x112 shape
    # new1_4(1)   checking
    std_landmarks = np.array([[85.49595008, 85.73911 + 30-15-15],
    [168.39541376, 84.30362432 + 30-15-15],
    [127.07756864, 136.76120384 + 30-15-15],
    [90.3434912, 174.31525376 + 30-15-15],
    [166.6558592, 172.99233344 + 30-15-15]], dtype=np.float32)  # right mouth corner

    landmarks = np.asarray(landmarks).reshape(5, 2)
    image_cropped, xform_coeff = align_and_crop_normal(image, landmarks, std_landmarks, (align_size, align_size))
    # transform landmark
    landmark_src = np.array(landmarks).astype(np.float32).reshape((5, 2))
    landmark = []
    for ls in landmark_src:
        landmark.append((*ls, 1))
    landmark_src = np.array(landmark).astype(np.float32).reshape((5, 3)).T
    landmark_src = np.dot(xform_coeff, landmark_src).T
    landmark = []
    for ls in landmark_src:
        landmark.append(ls[:2])
    landmark = np.array(landmark).reshape((-1)).astype(np.int32)
    return image_cropped, landmark

def prepare_img_stack_for_test(src_img, landmark, img_dim, need_augment=True, non_face_img=None):
    if landmark[0] == 0:
        if src_img.shape[0] != img_dim or src_img.shape[1] != img_dim:
            g_img = cv2.resize(src_img, (img_dim, img_dim))
        img_stack = g_img[..., :1]
        # cv2.imwrite('new_preprocess_fsbh_test_shifang.png', img_stack)
        img_stack = (img_stack.astype(np.uint8).astype(np.float32) - 127.5) / 127.5
        img_stack = img_stack.transpose((2, 0, 1))
        return img_stack  # shape: [1, img_dim, img_dim]
    
    # 根据关键点得到人脸框信息，并扩大1.8倍
    log_debug_info('src img first 10 elements', src_img.ravel()[0:10])
    bbox_from_lm = get_bbox_from_landmark(landmark)
    log_debug_info('bbox from landmarks', bbox_from_lm)
    # print("bbox_from_lm: ", bbox_from_lm)

    bbox = expand_bbox(np.array(bbox_from_lm).astype(np.int32), ratio=1.8,   #1.4,
                       max_w=src_img.shape[1], max_h=src_img.shape[0])
    # print("expand bbox: ", bbox)

    # 裁剪出人脸框内的图像
    g_img = src_img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

    if g_img is None or 0 in g_img.shape:
        return None
    log_debug_info('crop img first 10 elements', g_img.ravel()[0:10])
    #print('crop img first 10 elements', g_img.ravel()[0:10])

    # 更新裁剪后关键点
    landmark_crop = []
    j = 0
    for i in landmark:
        if j % 2 == 0:
            landmark_crop.append(i-bbox[0])
        else:
            landmark_crop.append(i-bbox[1])
        j = j+1
    # print("new landmark: ", landmark_crop)  

    # 直接彷射变换到256*256
    # g_img_resize, landmark_resize = align_and_crop(g_img, landmark_crop, align_size=256)
    g_img_resize, landmark_resize = align_and_crop_test(g_img, landmark_crop, align_size=256)
    log_debug_info('align and crop img first 10 elements', src_img.ravel()[0:10])
    # print('align and crop img first 10 elements', src_img.ravel()[0:10])

    img_stack = g_img_resize[..., :1]
    if need_augment:
        if np.random.random() < 0.5:
            img_stack = img_stack[:, ::-1, :]
        if np.random.random() < 0.5:
            img_stack = img_stack[::-1, :, :]
        if non_face_img is not None:  # this indicates current is not live samples, we can do any augments we like
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, alpha=random.uniform(0.5, 1.5))
            if np.random.random() < 0.5:
                img_stack = brightness_distortion(img_stack, beta=random.uniform(-32, 32))
            if np.random.random() < 0.4:  # slightly change pixels
                origin_type = img_stack.dtype
                img_stack = img_stack.astype(np.float32) + np.random.rand(*img_stack.shape) * 20 - 10
                img_stack[img_stack < 0] = 0
                img_stack[img_stack > 255] = 255
                img_stack = img_stack.astype(origin_type)

    # 保存为图片
    # img_stack_new = np.squeeze(img_stack)   # img_stack: 256*256*1 img_stack_new: 256*256
    # cv2.imwrite('new_preprocess_fsbh_test.png', img_stack_new)

    # 归一化，并转换通道hwc为chw
    img_stack = (img_stack.astype(np.uint8).astype(np.float32) - 127.5) / 127.5
    log_debug_info('normalized img first 10 elements', img_stack.ravel()[0:10])
    img_stack = img_stack.transpose((2, 0, 1))

    return img_stack  # shape: [1, img_dim, img_dim]

# def prepare_img_stack_same_process(src_img, landmark, img_dim, need_augment=True, non_face_img=None):
    
#     # 转化为单通道
#     img_stack = src_img[..., :1]
#     img_stack = np.squeeze(img_stack)

#     # 直接调用接口
#     '''
#         1、根据关键点得到人脸框信息，并扩大1.8倍数
#         2、crop出该人脸框内的图像，修改对应关键点
#         3、直接仿射变换到256*256
#         4、归一化，并转化通道hwc为chw
#     '''  
#     img_stack = align_and_crop_amba_cv28(img_stack, landmark, version=0)
#     # print("img_stack shape: ", img_stack.shape)

#     # 扩展为[1, img_dim, img_dim]
#     # img_stack = img_stack.unsqueeze(0)
#     # img_stack = torch.unsqueeze(img_stack, 0)
#     img_stack = np.expand_dims(img_stack, axis=0)

#     return img_stack


def get_fake_fp_landmark_tuples_from_list(fake_2d_list_fp, fake_3d_list_fp, ratio_3d: float, max_samples_each_list, verbose=True):
    assert 0.0 < ratio_3d <= 1.0, ratio_3d
    fk_tuples_2d = get_and_align_fp_landmark_tuples_from_fp_list(
        fake_2d_list_fp, max_samples_each_list, need_align=False, verbose=verbose)
    fk_tuples = get_and_align_fp_landmark_tuples_from_fp_list(
        fake_3d_list_fp, max_samples_each_list, need_align=False, verbose=verbose)

    np.random.shuffle(fk_tuples_2d)
    max_2d_len = min(len(fk_tuples_2d), int(len(fk_tuples) * (1.0-ratio_3d) / ratio_3d))
    if verbose:
        print('total 2d/3d fake samples: %d/%d, keep %d 2d samples.' % (len(fk_tuples_2d), len(fk_tuples), max_2d_len))
    fk_tuples += fk_tuples_2d[:max_2d_len]
    return fk_tuples  # it is a list, each element is (fp, landmark)


def get_fake_img_fp_twin(fake_img_fp):
    fake_dir, fake_fn = os.path.split(fake_img_fp)
    if '_0.png' in fake_fn:
        fake_fn_t = fake_fn.replace('_0.png', '_2.png')
    else:
        fake_fn_t = fake_fn.replace('_2.png', '_0.png')
    expected_fp = os.path.join(fake_dir, fake_fn_t)
    if os.path.exists(expected_fp):
        return expected_fp
    return fake_img_fp

import time

def get_fp_landmark_tuples_from_list(train_list_fp, fake_flag=True, verbose=True):
    fp_landmark_tuples = []
    file_fp = train_list_fp.split(',')
    for list_i in file_fp:
        with open(list_i, 'r') as list_f:
            for line in list_f:
                img_info = line.split(' ')
                label = img_info[1].strip('\n')
                if fake_flag==True:
                    tmp = '1'
                else:
                    tmp = '0'

                if label == tmp:
                    img_path_info = img_info[0].strip('\n')
                    if "shifang" in line:
                        # continue
                        img_path_info_tuples = (img_path_info, [0])                        
                        #print(line)
                    else:
                        img_path_landmark = img_path_info.split(';')
                        landmarks_str_list = img_path_landmark[2].split(',')
                        landmarks_list = [float(e) for e in landmarks_str_list]
                        img_path_info_tuples = (img_path_landmark[0], landmarks_list)
                        #print(line)
                    fp_landmark_tuples.append(img_path_info_tuples)
                
   
    return fp_landmark_tuples


class TrainingDataSet(torch_data.Dataset):
    def __init__(self, live_dict_list_fp, fake_2d_list_fp, fake_3d_list_fp, train_list_mix_fp, ratio_3d,
                 img_dim, max_samples_each_id, max_samples_each_list, need_augment=True):
        self.live_dict_list_fp, self.fake_2d_list_fp = live_dict_list_fp, fake_2d_list_fp
        self.fake_3d_list_fp, self.ratio_3d = fake_3d_list_fp, ratio_3d
        self.max_samples_each_id = max_samples_each_id
        self.max_samples_each_list = max_samples_each_list
        self.train_list_mix_fp = train_list_mix_fp    # 用于兼容格式不同的GRBD数据

        if self.train_list_mix_fp is not None:
            # 此处需要适配，在列表中将所有的假体与活体分开
            # live_fp_landmark_tuples和fake_fp_landmark_tuples：均为 list 结构，其中的每个元素为 (img_path, landmark) 形式的元组，其中的landmark为一个float格式的list
            self.live_fp_landmark_tuples = get_fp_landmark_tuples_from_list(self.train_list_mix_fp, fake_flag=False, verbose=True)
            self.fake_fp_landmark_tuples = get_fp_landmark_tuples_from_list(self.train_list_mix_fp, fake_flag=True, verbose=True)

        if self.live_dict_list_fp is not None:
            live_tmp_list = get_fp_landmark_tuples_from_json_list(self.live_dict_list_fp, self.max_samples_each_id, verbose=True)
            self.live_fp_landmark_tuples = live_tmp_list

        if self.fake_2d_list_fp and self.fake_3d_list_fp is not None:
            fake_tmp_list = get_fake_fp_landmark_tuples_from_list(self.fake_2d_list_fp, self.fake_3d_list_fp, self.ratio_3d, self.max_samples_each_list, verbose=True)
            self.fake_fp_landmark_tuples = fake_tmp_list

        self.live_ds_len = len(self.live_fp_landmark_tuples)
        self.fake_ds_len = len(self.fake_fp_landmark_tuples)
        print('total data set len: %d(live), %d(fake).' % (self.live_ds_len, self.fake_ds_len))
        self.img_dim = img_dim
        self.need_augment = need_augment
        self.img_nums_load = 0
    
    def shuffle(self, reload=False):
        if reload:
            print('reloading data set...')
            if self.train_list_mix_fp is not None:
                self.live_fp_landmark_tuples = get_fp_landmark_tuples_from_list(self.train_list_mix_fp, fake_flag=False, verbose=True)
                self.fake_fp_landmark_tuples = get_fp_landmark_tuples_from_list(self.train_list_mix_fp, fake_flag=True, verbose=True)
                self.live_ds_len = len(self.live_fp_landmark_tuples)
                self.fake_ds_len = len(self.fake_fp_landmark_tuples)
                print('total data set len: %d(live), %d(fake).' % (self.live_ds_len, self.fake_ds_len))

            if self.live_dict_list_fp is not None:             
                live_tmp_list = get_fp_landmark_tuples_from_json_list(self.live_dict_list_fp, self.max_samples_each_id, verbose=False)
                self.live_fp_landmark_tuples += live_tmp_list
                self.live_ds_len = len(self.live_fp_landmark_tuples)
                print('total data set len: %d(live).' % (self.live_ds_len))
            
            if self.live_dict_list_fp and self.fake_2d_list_fp and self.fake_3d_list_fp is not None:             
                fake_tmp_list = get_fake_fp_landmark_tuples_from_list(self.fake_2d_list_fp, self.fake_3d_list_fp, self.ratio_3d, self.max_samples_each_list, verbose=False)
                self.fake_fp_landmark_tuples += fake_tmp_list
                self.fake_ds_len = len(self.fake_fp_landmark_tuples)
                print('total data set len: %d(fake).' % (self.fake_ds_len))

        np.random.shuffle(self.live_fp_landmark_tuples)
        np.random.shuffle(self.fake_fp_landmark_tuples)

    def __len__(self):
        return max(self.live_ds_len, self.fake_ds_len)

    def __getitem__(self, item):

        # start_time = time.time()

        live_fp, live_landmark = self.live_fp_landmark_tuples[item % self.live_ds_len]
        fake_fp, fake_landmark = self.fake_fp_landmark_tuples[item % self.fake_ds_len]

        if 'dataWhouse' in live_fp:
            live_fp = live_fp.replace('dataWhouse', 'dataWarehouse')
        if '/storage-server5/dataWarehouse/lvmi/data_DM8/hefei_office' in live_fp:
            live_fp = live_fp.replace('/storage-server5/dataWarehouse/lvmi/data_DM8/hefei_office', '/storage-server5/dataWarehouse/lvmi/data_DM8/hefei_office/lvDM8_data01_20220315')
        live_img = cv2.imread(live_fp)
        # fake fp has 2 format: end with '_0.png' or '_2.png', they share the same content,
        # and '_0.png' is processed result of origin '_2.png'
        fake_img = cv2.imread(fake_fp)
        fake_twin_fp = get_fake_img_fp_twin(fake_fp)
        fake_twin_img = cv2.imread(fake_twin_fp)

        if live_img is None or fake_img is None or fake_twin_img is None:
            print('img read failed,', live_fp, fake_fp, fake_twin_fp)
            return self.__getitem__(item + 1)

        rd = np.random.random()
        # if rd < 0.33:
        if rd < 0.5:
            random_fake_img, random_fake_landmark = random_crop(
                fake_img, fake_landmark, dst_dim=self.img_dim * 2)

        elif rd < -1:    # 0.5   耗时太高，直接去掉
            # random_fake_img = get_occ_live_img_as_fake(live_img, live_landmark, fake_img)
            # random_fake_landmark = live_landmark.copy()
            random_fake_img = get_occ_live_img_as_fake(fake_img, fake_landmark, fake_img)
            random_fake_landmark = fake_landmark.copy()

        elif rd < 0.75:
            random_fake_img = random_bright_or_dark_landmark_area(fake_img, fake_landmark) 
            random_fake_landmark = fake_landmark.copy()

        else:
            # cv2.imwrite("random_bright_or_dark_before.png", live_img)
            random_fake_img = random_bright_or_dark_landmark_area(live_img, live_landmark)
            random_fake_landmark = live_landmark.copy()
            # cv2.imwrite("random_bright_or_dark_after.png", random_fake_img)

        random_fake_stack = prepare_img_stack(random_fake_img, landmark=random_fake_landmark, img_dim=self.img_dim, need_augment=self.need_augment)

        # live_img_stack = prepare_img_stack(live_img, landmark=live_landmark, img_dim=self.img_dim, need_augment=self.need_augment)
        rd = np.random.random()
        if 'C0' in live_fp and rd < 0.3:
            # glasses  
            live_img = random_bright_left_or_right_glasses_landmark_area(live_img, live_landmark)
        
        # 真人人脸随机裁剪, 没有多大意义
        # elif rd < 0.5:
        #     live_img, random_live_landmark = random_crop(live_img, live_landmark, dst_dim=self.img_dim * 2)
        #     live_landmark = random_live_landmark

        live_img_stack = prepare_img_stack(live_img, landmark=live_landmark, img_dim=self.img_dim, need_augment=self.need_augment)
        
        rd_w = max(random_fake_img.shape[1] - 40, random_fake_img.shape[1])
        rd_h = max(random_fake_img.shape[0] - 40, random_fake_img.shape[0])
        rd_xs, rd_ys = int(random.uniform(0, rd_w)), int(random.uniform(0, rd_h))
        fake_img_stack = prepare_img_stack(fake_img, landmark=fake_landmark, img_dim=self.img_dim, need_augment=self.need_augment, non_face_img=random_fake_img.copy()[rd_ys:rd_ys+rd_h, rd_xs:rd_xs+rd_w, :])
        fake_twin_img_stack = prepare_img_stack(fake_twin_img, landmark=fake_landmark, img_dim=self.img_dim, need_augment=False, non_face_img=None)

        if live_img_stack is None or fake_img_stack is None or \
                random_fake_stack is None or fake_twin_img_stack is None:
            print('build image stack failed.')
            return self.__getitem__(item + 1)

        # end_time = time.time()
        # self.img_nums_load += 1
        # print("====> time cost: {:.2f}s".format(end_time - start_time))
        # print(self.img_nums_load)

        return live_img_stack, fake_img_stack, random_fake_stack, fake_twin_img_stack


def get_training_data_loader(
        live_dict_list_fp, fake_2d_list_fp, fake_3d_list_fp, train_list_mix_fp, ratio_3d, img_dim, batch_size, num_workers,
        max_samples_each_id=30, max_samples_each_list=5000, ddp=False, need_augment=True):
    ds = TrainingDataSet(
        live_dict_list_fp, fake_2d_list_fp, fake_3d_list_fp, train_list_mix_fp, ratio_3d, img_dim, max_samples_each_id,
        max_samples_each_list, need_augment
    )
    if not ddp:
        dl = torch_data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return ds, dl
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
    dl = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True)
    return ds, dl


class TestingDataSet(torch_data.Dataset):
    def __init__(self, fp_list_fp, img_dim, dict_src: bool, fake_flag=True, datalist_normal=True):
        if dict_src:
            self.fp_landmark_tuples = get_fp_landmark_tuples_from_json_list(fp_list_fp, 99999999)
        else:

            # 以下两句待兼容
            if datalist_normal == True:
                # DM8格式数据
                self.fp_landmark_tuples = get_and_align_fp_landmark_tuples_from_fp_list_test(fp_list_fp, 99999999, need_align=False)
            else:
                # RGBD脏乱的格式数据
                self.fp_landmark_tuples = get_fp_landmark_tuples_from_list(fp_list_fp, fake_flag=fake_flag, verbose=True)

        self.ds_len = len(self.fp_landmark_tuples)
        print('total data set len: %d.' % self.ds_len)
        self.img_dim = img_dim

    def __len__(self):
        return self.ds_len

    def __getitem__(self, item):
        fp, landmark = self.fp_landmark_tuples[item]
        img = cv2.imread(fp)
        if img is None:
            print('img read failed.')
            return self.__getitem__(item + 1)

        # 原始方式
        # img_stack = prepare_img_stack(img, landmark=landmark, img_dim=self.img_dim, need_augment=False)

        # python版本修改后预处理
        img_stack = prepare_img_stack_for_test(img, landmark=landmark, img_dim=self.img_dim, need_augment=False)
        
        # c++版本修改后预处理
        # img_stack = prepare_img_stack_same_process(img, landmark=landmark, img_dim=self.img_dim, need_augment=False)

        return img_stack, fp

def get_testing_data_loader(fp_list_fp, batch_size, num_workers, img_dim, dict_src: bool, ddp=False, fake_flag=True, datalist_normal=True):
    ds = TestingDataSet(fp_list_fp, img_dim, dict_src, fake_flag, datalist_normal)
    if not ddp:
        dl = torch_data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dl
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    dl = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return dl

class SamplesCache:
    def __init__(self, max_num_caches: int, num_sample_types: int, sample_shape: list, device='cuda'):
        assert max_num_caches > 0 and num_sample_types > 0, (max_num_caches, num_sample_types)
        assert len(sample_shape) == 3, sample_shape
        assert (np.array(sample_shape) > 0).all(), sample_shape

        self.max_num_caches, self.num_sample_types = max_num_caches, num_sample_types
        self.sample_shape, self.device = sample_shape, device

        self.caches = [[] for _ in range(self.num_sample_types)]
        self.cache_nums = [0 for _ in range(len(self.caches))]

    def set_caches(self, samples: list):
        assert len(samples) == self.num_sample_types, len(samples)
        for k in range(self.num_sample_types):
            c_sample = samples[k]
            assert isinstance(c_sample, torch.Tensor), type(c_sample)
            assert len(c_sample.shape) == 4, c_sample.shape
            assert c_sample.shape[1:] == torch.Size(self.sample_shape), (c_sample.shape, self.sample_shape)
            c_idx = self.cache_nums[k] % self.max_num_caches
            if len(self.caches[k]) - 1 < c_idx:
                self.caches[k] += [c_sample.detach().clone()]
            else:
                self.caches[k][c_idx] = c_sample.detach().clone()
            self.cache_nums[k] += 1

    def get_caches(self):
        if len(self.caches[0]) == 0:
            print('no cache now.')
            return None
        return self.caches












