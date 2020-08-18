# UTILITY-FUNCTIONS (PROCESS DATA)

from __future__ import print_function

import numpy as np
from src.utils import *

__all__ = ["read_train_data", "read_train_data_multi"]


def read_train_data(filename2read, dim_p, h_prev, h_post, idx_f_use, idx_r_use, sample_ratio, use_x_sp=False,
                    use_y_sp=False):
    """ Reads train-data (single). """
    idx_f_use = make_numpy_array(idx_f_use, keep_1dim=True)
    idx_r_use = make_numpy_array(idx_r_use, keep_1dim=True)

    len_filename = len(filename2read)
    data_size = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
        f_train_in = data_read_tmp[()]['data_f']
        data_size = data_size + f_train_in.shape[0]

    print("data_size (before): {:d}".format(data_size))

    dim_x, dim_y = dim_p * h_prev, dim_p * h_post
    dim_x_3 = 3 * h_prev
    if use_y_sp:
        dim_y_3 = 3 * (h_post + 1)
    else:
        dim_y_3 = 3 * (h_post + 2)
    dim_f = idx_f_use.shape[0] if len(idx_f_use) > 0 else 0
    dim_r = idx_r_use.shape[0] if len(idx_r_use) > 0 else 0

    idx_xin_tmp, idx_yin_tmp = np.arange(0, dim_x_3), np.arange(0, dim_y_3)
    if use_y_sp:
        idx_y0 = np.arange(0, dim_p * h_post)
        idx_y1 = np.arange(dim_p, dim_p * (h_post + 1))
    else:
        idx_y0 = np.arange(dim_p, dim_p * (h_post + 1))
        idx_y1 = np.arange(dim_p * 2, dim_p * (h_post + 2))

    h_prev_ref, h_post_ref = dim_x_3, dim_y_3

    x_train = np.zeros((data_size, dim_x), dtype=np.float32)
    y0_train = np.zeros((data_size, dim_y), dtype=np.float32)
    y1_train = np.zeros((data_size, dim_y), dtype=np.float32)
    f_train = np.zeros((data_size, dim_f), dtype=np.float32) if dim_f > 0 else []
    r_train = np.zeros((data_size, dim_r), dtype=np.float32) if dim_r > 0 else []

    cnt_data = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)

        x_train_in = data_read_tmp[()]['data_x_sp'] if use_x_sp else data_read_tmp[()]['data_x']
        y_train_in = data_read_tmp[()]['data_y_sp'] if use_y_sp else data_read_tmp[()]['data_y']

        if nidx_d == 0:
            h_prev_ref, h_post_ref = x_train_in.shape[1], y_train_in.shape[1]

        if dim_p == 2:
            idx_xin_tmp = np.setdiff1d(idx_xin_tmp, np.arange(2, h_prev_ref, 3))
            idx_yin_tmp = np.setdiff1d(idx_yin_tmp, np.arange(2, h_post_ref, 3))

        x_train_in = x_train_in[:, idx_xin_tmp]
        y_train_in = y_train_in[:, idx_yin_tmp]

        if dim_f > 0:
            f_train_in = data_read_tmp[()]['data_f']
            f_train_in = f_train_in[:, idx_f_use]

        if dim_r > 0:
            r_train_in = data_read_tmp[()]['data_r']
            r_train_in = r_train_in[:, idx_r_use]

        # Update
        len_before = x_train_in.shape[0]
        idx_rand_tmp_ = np.random.permutation(len_before)
        len_after = int(sample_ratio[nidx_d] * len_before)
        idx_rand_tmp = idx_rand_tmp_[np.arange(0, len_after)]

        idx_update_tmp = np.arange(cnt_data, cnt_data + len_after)
        x_train[idx_update_tmp, :] = x_train_in[idx_rand_tmp, :]
        y_train_in_tmp = y_train_in[idx_rand_tmp, :]

        y0_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y0]
        y1_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y1]

        if dim_f > 0:
            f_train[idx_update_tmp, :] = f_train_in[idx_rand_tmp, :]

        if dim_r > 0:
            r_train[idx_update_tmp, :] = r_train_in[idx_rand_tmp, :]

        cnt_data = cnt_data + len_after

    print("data_size (after): {:d}".format(cnt_data))

    idx_update = np.arange(0, cnt_data)
    x_train = x_train[idx_update, :]
    y0_train = y0_train[idx_update, :]
    y1_train = y1_train[idx_update, :]

    if dim_f > 0:
        f_train = f_train[idx_update, :]
    else:
        f_train = []

    if dim_r > 0:
        r_train = r_train[idx_update, :]

    return x_train, y0_train, y1_train, f_train, r_train


def read_train_data_multi(filename2read, dim_p, h_prev, h_post, idx_f_use, idx_r_use, num_near, sample_ratio=1.0,
                          use_x_sp=True, use_y_sp=True):
    """ Reads train-data (multiple). """

    idx_f_use = make_numpy_array(idx_f_use, keep_1dim=True)
    idx_r_use = make_numpy_array(idx_r_use, keep_1dim=True)

    len_filename = len(filename2read)
    data_size = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)
        f_train_in = data_read_tmp[()]['data_f']
        data_size = data_size + f_train_in.shape[0]

    print("data_size (before): {:d}".format(data_size))

    dim_x, dim_y = dim_p * h_prev, dim_p * h_post
    dim_x_3 = 3 * h_prev
    if use_y_sp:
        dim_y_3 = 3 * (h_post + 1)
    else:
        dim_y_3 = 3 * (h_post + 2)
    dim_f = idx_f_use.shape[0] if len(idx_f_use) > 0 else 0
    dim_r = idx_r_use.shape[0] if len(idx_r_use) > 0 else 0

    idx_xin_tmp, idx_yin_tmp = np.arange(0, dim_x_3), np.arange(0, dim_y_3)
    if use_y_sp:
        idx_y0 = np.arange(0, dim_p * h_post)
        idx_y1 = np.arange(dim_p, dim_p * (h_post + 1))
    else:
        idx_y0 = np.arange(dim_p, dim_p * (h_post + 1))
        idx_y1 = np.arange(dim_p * 2, dim_p * (h_post + 2))

    h_prev_ref, h_post_ref = dim_x_3, dim_y_3

    x_train = np.zeros((data_size, dim_x), dtype=np.float32)
    y0_train = np.zeros((data_size, dim_y), dtype=np.float32)
    y1_train = np.zeros((data_size, dim_y), dtype=np.float32)
    f_train = np.zeros((data_size, dim_f), dtype=np.float32) if dim_f > 0 else []
    r_train = np.zeros((data_size, dim_r), dtype=np.float32) if dim_r > 0 else []

    xnear_train = np.zeros((data_size, num_near, dim_x), dtype=np.float32)
    y0near_train = np.zeros((data_size, num_near, dim_y), dtype=np.float32)
    y1near_train = np.zeros((data_size, num_near, dim_y), dtype=np.float32)
    fnear_train = np.zeros((data_size, num_near, dim_f), dtype=np.float32) if dim_f > 0 else []
    rnear_train = np.zeros((data_size, num_near, dim_r), dtype=np.float32) if dim_r > 0 else []

    cnt_data = 0
    for nidx_d in range(0, len_filename):
        filename2read_sel = filename2read[nidx_d]
        data_read_tmp = np.load(filename2read_sel, allow_pickle=True)

        x_train_in = data_read_tmp[()]['data_x_sp'] if use_x_sp else data_read_tmp[()]['data_x']
        y_train_in = data_read_tmp[()]['data_y_sp'] if use_y_sp else data_read_tmp[()]['data_y']
        xnear_train_in = data_read_tmp[()]['data_xnear_sp'] if use_x_sp else data_read_tmp[()]['data_xnear']
        ynear_train_in = data_read_tmp[()]['data_ynear_sp'] if use_x_sp else data_read_tmp[()]['data_ynear']

        if nidx_d == 0:
            h_prev_ref, h_post_ref = x_train_in.shape[1], y_train_in.shape[1]

        if dim_p == 2:
            idx_xin_tmp = np.setdiff1d(idx_xin_tmp, np.arange(2, h_prev_ref, 3))
            idx_yin_tmp = np.setdiff1d(idx_yin_tmp, np.arange(2, h_post_ref, 3))

        x_train_in = x_train_in[:, idx_xin_tmp]
        y_train_in = y_train_in[:, idx_yin_tmp]
        xnear_train_in = xnear_train_in[:, :, idx_xin_tmp]
        # xnear_train_in = xnear_train_in.reshape(-1, xnear_train_in.shape[1] * xnear_train_in.shape[2])
        ynear_train_in = ynear_train_in[:, :, idx_yin_tmp]
        # ynear_train_in = ynear_train_in.reshape(-1, ynear_train_in.shape[1] * ynear_train_in.shape[2])

        if dim_f > 0:
            f_train_in = data_read_tmp[()]['data_f']
            f_train_in = f_train_in[:, idx_f_use]
            fnear_train_in = data_read_tmp[()]['data_fnear']
            fnear_train_in = fnear_train_in[:, :, idx_f_use]

        if dim_r > 0:
            r_train_in = data_read_tmp[()]['data_r']
            r_train_in = r_train_in[:, idx_r_use]
            rnear_train_in = data_read_tmp[()]['data_rnear']
            rnear_train_in = rnear_train_in[:, :, idx_r_use]

        # Update
        len_before = x_train_in.shape[0]
        idx_rand_tmp_ = np.random.permutation(len_before)
        len_after = int(sample_ratio[nidx_d] * len_before)
        idx_rand_tmp = idx_rand_tmp_[np.arange(0, len_after)]

        idx_update_tmp = np.arange(cnt_data, cnt_data + len_after)
        x_train[idx_update_tmp, :] = x_train_in[idx_rand_tmp, :]
        y_train_in_tmp = y_train_in[idx_rand_tmp, :]
        y0_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y0]
        y1_train[idx_update_tmp, :] = y_train_in_tmp[:, idx_y1]

        xnear_train[idx_update_tmp, :, :] = xnear_train_in[idx_rand_tmp, :, :]
        ynear_train_in_tmp = ynear_train_in[idx_rand_tmp, :, :]
        y0near_train[idx_update_tmp, :, :] = ynear_train_in_tmp[:, :, idx_y0]
        y1near_train[idx_update_tmp, :, :] = ynear_train_in_tmp[:, :, idx_y1]

        if dim_f > 0:
            f_train[idx_update_tmp, :] = f_train_in[idx_rand_tmp, :]
            fnear_train[idx_update_tmp, :, :] = fnear_train_in[idx_rand_tmp, :, :]

        if dim_r > 0:
            r_train[idx_update_tmp, :] = r_train_in[idx_rand_tmp, :]
            rnear_train[idx_update_tmp, :, :] = rnear_train_in[idx_rand_tmp, :, :]

        cnt_data = cnt_data + len_after

    print("data_size (after): {:d}".format(cnt_data))

    idx_update = np.arange(0, cnt_data)
    x_train = x_train[idx_update, :]
    y0_train = y0_train[idx_update, :]
    y1_train = y1_train[idx_update, :]
    xnear_train = xnear_train[idx_update, :, :]
    y0near_train = y0near_train[idx_update, :, :]
    y1near_train = y1near_train[idx_update, :, :]

    if dim_f > 0:
        f_train = f_train[idx_update, :]
        fnear_train = fnear_train[idx_update, :, :]

    if dim_r > 0:
        r_train = r_train[idx_update, :]
        rnear_train = rnear_train[idx_update, :]

    return x_train, y0_train, y1_train, f_train, r_train, xnear_train, y0near_train, y1near_train, fnear_train, \
           rnear_train
