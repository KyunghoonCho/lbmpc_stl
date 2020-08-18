# TRAIN DATA

import numpy as np
import math
from src.utils import *
from src.utils_sim import *
from src.utils_readdata import *


class TrainData(object):
    def __init__(self, filename2read, sample_ratio_data, sample_ratio_train, sample_ratio_test, num_train, num_test,
                 h_prev, h_post, dim_p, idx_f_use, idx_r_use, batch_size, use_x_sp=True, use_y_sp=True, dis_f=False,
                 load_multi=False, num_near=6):
        self.filename2read = filename2read
        self.sample_ratio_data = sample_ratio_data
        self.sample_ratio_train, self.sample_ratio_test = sample_ratio_train, sample_ratio_test
        self.num_train, self.num_test = num_train, num_test
        self.dim_p, self.idx_f_use, self.idx_r_use = dim_p, idx_f_use, idx_r_use
        self.batch_size = batch_size
        self.use_x_sp, self.use_y_sp = use_x_sp, use_y_sp
        self.dis_f = dis_f
        self.load_multi = load_multi
        self.num_near = num_near

        self.h_prev = int(h_prev / 2) if use_x_sp else h_prev
        self.h_post = int(h_post / 2) if use_y_sp else h_post
        # self.h_post = self.h_post + 1  # Extra time-step for the first input

        # Data
        self.xd_min, self.xd_max = [], []
        self.yd_min, self.yd_max = [], []
        self.f_min, self.f_max = [], []
        self.r_min, self.r_max = [], []

        self.x_train, self.f_train = [], []
        self.y0_train, self.y1_train, self.r_train = [], [], []
        self.c_train = []
        self.x_train_n, self.f_train_n = [], []
        self.y0_train_n, self.y1_train_n, self.r_train_n = [], [], []  # Normalized target data (train)
        self.c_train_n = []

        self.x_test, self.f_test = [], []
        self.y0_test, self.y1_test, self.r_test = [], [], []
        self.c_test = []
        self.x_test_n, self.f_test_n = [], []
        self.y0_test_n, self.y1_test_n, self.r_test_n = [], [], []  # Normalized target data (test)
        self.c_test_n = []

        self.n_traindata, self.n_testdata = 0, 0
        self.idx_traindata, self.idx_testdata = [], []

        self.xd_train_mean, self.xd_train_std, self.yd_train_mean, self.yd_train_std = [], [], [], []
        self.x_train_mean, self.x_train_std = [], []
        self.y_train_mean, self.y_train_std = [], []
        self.f_train_mean, self.f_train_std, self.r_train_mean, self.r_train_std = [], [], [], []
        self.c_train_mean, self.c_train_std = [], []

        # Training
        self.n_batch_train, self.n_batch_test = 0, 0

        # Multiple case
        if self.load_multi:
            self.xnear_train, self.xnear_train_n = [], []
            self.y0near_train, self.y1near_train, self.y0near_train_n, self.y1near_train_n = [], [], [], []
            self.fnear_train, self.fnear_train_n = [], []
            self.rnear_train, self.rnear_train_n = [], []

            self.xnear_test, self.xnear_test_n = [], []
            self.y0near_test, self.y1near_test, self.y0near_test_n, self.y1near_test_n = [], [], [], []
            self.fnear_test, self.fnear_test_n = [], []
            self.rnear_test, self.rnear_test_n = [], []

    def processing(self):
        """ Processes data. """
        if self.load_multi:
            self.processing_multi()
        else:
            self.processing_single()

    def processing_single(self):
        """ Processes data (single). """
        x_data, y0_data, y1_data, f_data, r_data = read_train_data(self.filename2read, self.dim_p, self.h_prev,
                                                                   self.h_post, self.idx_f_use, self.idx_r_use,
                                                                   self.sample_ratio_data, use_x_sp=self.use_x_sp,
                                                                   use_y_sp=self.use_y_sp)

        if self.dis_f:
            f_data = get_discretized_f(f_data, self.idx_f_use)

        self.processing_common(x_data, y0_data, y1_data, f_data, r_data)

    def processing_multi(self):
        """ Processes data (multiple). """
        x_data, y0_data, y1_data, f_data, r_data, xnear_data, y0near_data, y1near_data, fnear_data, rnear_data = \
            read_train_data_multi(self.filename2read, self.dim_p, self.h_prev, self.h_post, self.idx_f_use,
                                  self.idx_r_use, self.num_near, self.sample_ratio_data, use_x_sp=self.use_x_sp,
                                  use_y_sp=self.use_y_sp)

        if self.dis_f:
            f_data = get_discretized_f(f_data, self.idx_f_use)
            fnear_data_list = np.split(fnear_data, self.num_near, axis=1)
            for nidx_n in range(0, self.num_near):
                fnear_data_tmp = np.squeeze(fnear_data_list[nidx_n])
                fnear_data_list[nidx_n] = get_discretized_f(fnear_data_tmp, self.idx_f_use)
            fnear_data = np.stack(fnear_data_list, axis=1)

        self.processing_common(x_data, y0_data, y1_data, f_data, r_data)

        # # Normalize w.r.t. lanewidth
        # if len(self.idx_r_use) > 0:
        #     lw_data = fnear_data[:, :, 3]
        #     for nidx_r in range(0, len(self.idx_r_use)):
        #         if self.idx_r_use[nidx_r] < 4:
        #             rnear_data[:, :, nidx_r] = rnear_data[:, :, nidx_r] / lw_data

        # Set train-data (near)
        self.xnear_train = xnear_data[self.idx_traindata, :, :]
        self.y0near_train = y0near_data[self.idx_traindata, :, :]
        self.y1near_train = y1near_data[self.idx_traindata, :, :]
        if len(self.idx_f_use) > 0:
            self.fnear_train = fnear_data[self.idx_traindata, :, :]
        if len(self.idx_r_use) > 0:
            self.rnear_train = rnear_data[self.idx_traindata, :, :]

        # Set test-data (near)
        self.xnear_test = xnear_data[self.idx_testdata, :, :]
        self.y0near_test = y0near_data[self.idx_testdata, :, :]
        self.y1near_test = y1near_data[self.idx_testdata, :, :]
        if len(self.idx_f_use) > 0:
            self.fnear_test = fnear_data[self.idx_testdata, :, :]
        if len(self.idx_r_use) > 0:
            self.rnear_test = rnear_data[self.idx_testdata, :, :]

        # Normalize (near)
        xnear_train_list = np.split(self.xnear_train, self.num_near, axis=1)
        y0near_train_list = np.split(self.y0near_train, self.num_near, axis=1)
        y1near_train_list = np.split(self.y1near_train, self.num_near, axis=1)
        xnear_test_list = np.split(self.xnear_test, self.num_near, axis=1)
        y0near_test_list = np.split(self.y0near_test, self.num_near, axis=1)
        y1near_test_list = np.split(self.y1near_test, self.num_near, axis=1)

        xnear_train_list_n, y0near_train_list_n, y1near_train_list_n = [], [], []
        xnear_test_list_n, y0near_test_list_n, y1near_test_list_n = [], [], []
        for nidx_n in range(0, self.num_near):
            xnear_train_tmp = np.squeeze(xnear_train_list[nidx_n])
            y0near_train_tmp = np.squeeze(y0near_train_list[nidx_n])
            y1near_train_tmp = np.squeeze(y1near_train_list[nidx_n])
            xnear_test_tmp = np.squeeze(xnear_test_list[nidx_n])
            y0near_test_tmp = np.squeeze(y0near_test_list[nidx_n])
            y1near_test_tmp = np.squeeze(y1near_test_list[nidx_n])

            xnear_train_tmp_n = normalize_data_wrt_mean_scale(xnear_train_tmp, self.x_train_mean, self.x_train_std)
            y0near_train_tmp_n = normalize_data_wrt_mean_scale(y0near_train_tmp, self.y_train_mean, self.y_train_std)
            y1near_train_tmp_n = normalize_data_wrt_mean_scale(y1near_train_tmp, self.y_train_mean, self.y_train_std)
            xnear_test_tmp_n = normalize_data_wrt_mean_scale(xnear_test_tmp, self.x_train_mean, self.x_train_std)
            y0near_test_tmp_n = normalize_data_wrt_mean_scale(y0near_test_tmp, self.y_train_mean, self.y_train_std)
            y1near_test_tmp_n = normalize_data_wrt_mean_scale(y1near_test_tmp, self.y_train_mean, self.y_train_std)

            xnear_train_list_n.append(xnear_train_tmp_n)
            y0near_train_list_n.append(y0near_train_tmp_n)
            y1near_train_list_n.append(y1near_train_tmp_n)
            xnear_test_list_n.append(xnear_test_tmp_n)
            y0near_test_list_n.append(y0near_test_tmp_n)
            y1near_test_list_n.append(y1near_test_tmp_n)

        self.xnear_train_n = np.stack(xnear_train_list_n, axis=1)
        self.y0near_train_n = np.stack(y0near_train_list_n, axis=1)
        self.y1near_train_n = np.stack(y1near_train_list_n, axis=1)
        self.xnear_test_n = np.stack(xnear_test_list_n, axis=1)
        self.y0near_test_n = np.stack(y0near_test_list_n, axis=1)
        self.y1near_test_n = np.stack(y1near_test_list_n, axis=1)

        if len(self.idx_f_use) > 0:
            fnear_train_list = np.split(self.fnear_train, self.num_near, axis=1)
            fnear_test_list = np.split(self.fnear_test, self.num_near, axis=1)

            fnear_train_list_n, fnear_test_list_n = [], []
            for nidx_n in range(0, self.num_near):
                fnear_train_tmp = np.squeeze(fnear_train_list[nidx_n])
                fnear_test_tmp = np.squeeze(fnear_test_list[nidx_n])
                fnear_train_tmp_n = normalize_data_wrt_mean_scale(fnear_train_tmp, self.f_train_mean, self.f_train_std)
                fnear_test_tmp_n = normalize_data_wrt_mean_scale(fnear_test_tmp, self.f_train_mean, self.f_train_std)
                fnear_train_list_n.append(fnear_train_tmp_n)
                fnear_test_list_n.append(fnear_test_tmp_n)

            self.fnear_train_n = np.stack(fnear_train_list_n, axis=1)
            self.fnear_test_n = np.stack(fnear_test_list_n, axis=1)

        if len(self.idx_r_use) > 0:
            rnear_train_list = np.split(self.rnear_train, self.num_near, axis=1)
            rnear_test_list = np.split(self.rnear_test, self.num_near, axis=1)

            rnear_train_list_n, rnear_test_list_n = [], []
            for nidx_n in range(0, self.num_near):
                rnear_train_tmp = np.squeeze(rnear_train_list[nidx_n])
                rnear_test_tmp = np.squeeze(rnear_test_list[nidx_n])
                rnear_train_tmp_n = normalize_data_wrt_mean_scale(rnear_train_tmp, self.r_train_mean, self.r_train_std)
                rnear_test_tmp_n = normalize_data_wrt_mean_scale(rnear_test_tmp, self.r_train_mean, self.r_train_std)
                rnear_train_list_n.append(rnear_train_tmp_n)
                rnear_test_list_n.append(rnear_test_tmp_n)

            self.rnear_train_n = np.stack(rnear_train_list_n, axis=1)
            self.rnear_test_n = np.stack(rnear_test_list_n, axis=1)

        # Reshape (near)
        self.xnear_train = np.reshape(self.xnear_train, (self.n_traindata, self.num_near, self.h_prev, self.dim_p))
        self.y0near_train = np.reshape(self.y0near_train, (self.n_traindata, self.num_near, self.h_post, self.dim_p))
        self.y1near_train = np.reshape(self.y1near_train, (self.n_traindata, self.num_near, self.h_post, self.dim_p))
        self.xnear_train_n = np.reshape(self.xnear_train_n, (self.n_traindata, self.num_near, self.h_prev, self.dim_p))
        self.y0near_train_n = np.reshape(self.y0near_train_n, (self.n_traindata, self.num_near, self.h_post,
                                                               self.dim_p))
        self.y1near_train_n = np.reshape(self.y1near_train_n, (self.n_traindata, self.num_near, self.h_post,
                                                               self.dim_p))

        self.xnear_test = np.reshape(self.xnear_test, (self.n_testdata, self.num_near, self.h_prev, self.dim_p))
        self.y0near_test = np.reshape(self.y0near_test, (self.n_testdata, self.num_near, self.h_post, self.dim_p))
        self.y1near_test = np.reshape(self.y1near_test, (self.n_testdata, self.num_near, self.h_post, self.dim_p))
        self.xnear_test_n = np.reshape(self.xnear_test_n, (self.n_testdata, self.num_near, self.h_prev, self.dim_p))
        self.y0near_test_n = np.reshape(self.y0near_test_n, (self.n_testdata, self.num_near, self.h_post, self.dim_p))
        self.y1near_test_n = np.reshape(self.y1near_test_n, (self.n_testdata, self.num_near, self.h_post, self.dim_p))

    def processing_common(self, x_data, y0_data, y1_data, f_data, r_data):
        """ Processes data (common). """

        x_data = x_data.astype(np.float32)
        y0_data = y0_data.astype(np.float32)
        y1_data = y1_data.astype(np.float32)
        if len(self.idx_f_use) > 0:
            f_data = f_data.astype(np.float32)
        if len(self.idx_r_use) > 0:
            r_data = r_data.astype(np.float32)

            # # Normalize w.r.t. lanewidth
            # lw_data = f_data[:, 3]
            # for nidx_r in range(0, len(self.idx_r_use)):
            #     if self.idx_r_use[nidx_r] < 4:
            #         r_data[:, nidx_r] = r_data[:, nidx_r] / lw_data

        idx_ch_tmp = range(0, (self.dim_p * self.h_post), self.dim_p)
        idx_cv_tmp = range(1, (self.dim_p * self.h_post), self.dim_p)
        c_h_data_ = y1_data[:, idx_ch_tmp]
        c_v_data_ = y1_data[:, idx_cv_tmp]
        c_h_data_mean = np.mean(c_h_data_, axis=1)
        c_v_data_mean = np.mean(c_v_data_, axis=1)
        c_h_data_mean_r = np.reshape(c_h_data_mean, (-1, 1))
        c_v_data_mean_r = np.reshape(c_v_data_mean, (-1, 1))
        c_data = np.concatenate((c_h_data_mean_r, c_v_data_mean_r), axis=1)

        x_data_r = np.reshape(x_data, (-1, self.dim_p))
        y1_data_r = np.reshape(y1_data, (-1, self.dim_p))
        self.xd_min, self.xd_max = np.amin(x_data_r, axis=0), np.amax(x_data_r, axis=0)
        self.yd_min, self.yd_max = np.amin(y1_data_r, axis=0), np.amax(y1_data_r, axis=0)
        if len(self.idx_f_use) > 0:
            self.f_min, self.f_max = np.amin(f_data, axis=0), np.amax(f_data, axis=0)
        if len(self.idx_r_use) > 0:
            self.r_min, self.r_max = np.amin(r_data, axis=0), np.amax(r_data, axis=0)

        n_data = x_data.shape[0]
        idx_data_random = np.random.permutation(n_data)

        # Set indexes for 'train' & 'test'
        if self.num_train > 0:
            self.n_traindata = self.num_train
        else:
            self.n_traindata = int(n_data * self.sample_ratio_train)
        if self.num_test > 0:
            self.n_testdata = self.num_test
        else:
            self.n_testdata = int(n_data * self.sample_ratio_test)
        self.idx_traindata = idx_data_random[np.arange(0, self.n_traindata)]
        self.idx_testdata = idx_data_random[np.arange(self.n_traindata, self.n_traindata + self.n_testdata)]

        # Set train-data
        self.x_train = x_data[self.idx_traindata, :]
        self.y0_train = y0_data[self.idx_traindata, :]
        self.y1_train = y1_data[self.idx_traindata, :]
        if len(self.idx_f_use) > 0:
            self.f_train = f_data[self.idx_traindata, :]
        if len(self.idx_r_use) > 0:
            self.r_train = r_data[self.idx_traindata, :]
        self.c_train = c_data[self.idx_traindata, :]

        # Set test-data
        self.x_test = x_data[self.idx_testdata, :]
        self.y0_test = y0_data[self.idx_testdata, :]
        self.y1_test = y1_data[self.idx_testdata, :]
        if len(self.idx_f_use) > 0:
            self.f_test = f_data[self.idx_testdata, :]
        if len(self.idx_r_use) > 0:
            self.r_test = r_data[self.idx_testdata, :]
        self.c_test = c_data[self.idx_testdata, :]

        # Data mean & scale
        x_train_r = np.reshape(self.x_train, (-1, self.dim_p))
        y_train_r = np.reshape(self.y1_train, (-1, self.dim_p))
        _, self.xd_train_mean, self.xd_train_std = normalize_data(x_train_r)
        _, self.yd_train_mean, self.yd_train_std = normalize_data(y_train_r)

        self.x_train_mean = np.tile(self.xd_train_mean, self.h_prev)
        self.x_train_std = np.tile(self.xd_train_std, self.h_prev)
        self.y_train_mean = np.tile(self.yd_train_mean, self.h_post)
        self.y_train_std = np.tile(self.yd_train_std, self.h_post)
        self.x_train_mean = self.x_train_mean.astype(np.float32)
        self.x_train_std = self.x_train_std.astype(np.float32)
        self.y_train_mean = self.y_train_mean.astype(np.float32)
        self.y_train_std = self.y_train_std.astype(np.float32)
        if len(self.idx_f_use) > 0:
            _, self.f_train_mean, self.f_train_std = normalize_data(self.f_train)
            self.f_train_mean = self.f_train_mean.astype(np.float32)
            self.f_train_std = self.f_train_std.astype(np.float32)
        if len(self.idx_r_use) > 0:
            _, self.r_train_mean, self.r_train_std = normalize_data(self.r_train)
            self.r_train_mean = self.r_train_mean.astype(np.float32)
            self.r_train_std = self.r_train_std.astype(np.float32)
        _, self.c_train_mean, self.c_train_std = normalize_data(self.c_train)

        # Normalize
        self.x_train_n = normalize_data_wrt_mean_scale(self.x_train, self.x_train_mean, self.x_train_std)
        self.y0_train_n = normalize_data_wrt_mean_scale(self.y0_train, self.y_train_mean, self.y_train_std)
        self.y1_train_n = normalize_data_wrt_mean_scale(self.y1_train, self.y_train_mean, self.y_train_std)

        self.x_test_n = normalize_data_wrt_mean_scale(self.x_test, self.x_train_mean, self.x_train_std)
        self.y0_test_n = normalize_data_wrt_mean_scale(self.y0_test, self.y_train_mean, self.y_train_std)
        self.y1_test_n = normalize_data_wrt_mean_scale(self.y1_test, self.y_train_mean, self.y_train_std)

        if len(self.idx_f_use) > 0:
            self.f_train_n = normalize_data_wrt_mean_scale(self.f_train, self.f_train_mean, self.f_train_std)
            self.f_test_n = normalize_data_wrt_mean_scale(self.f_test, self.f_train_mean, self.f_train_std)

        if len(self.idx_r_use) > 0:
            self.r_train_n, self.r_train_mean, self.r_train_std = normalize_data(self.r_train)
            self.r_test_n = normalize_data_wrt_mean_scale(self.r_test, self.r_train_mean, self.r_train_std)

        self.c_train_n = normalize_data_wrt_mean_scale(self.c_train, self.c_train_mean, self.c_train_std)
        self.c_test_n = normalize_data_wrt_mean_scale(self.c_test, self.c_train_mean, self.c_train_std)

        # Reshape
        self.x_train = np.reshape(self.x_train, (self.n_traindata, self.h_prev, self.dim_p))
        self.y0_train = np.reshape(self.y0_train, (self.n_traindata, self.h_post, self.dim_p))
        self.y1_train = np.reshape(self.y1_train, (self.n_traindata, self.h_post, self.dim_p))
        self.x_train_n = np.reshape(self.x_train_n, (self.n_traindata, self.h_prev, self.dim_p))
        self.y0_train_n = np.reshape(self.y0_train_n, (self.n_traindata, self.h_post, self.dim_p))
        self.y1_train_n = np.reshape(self.y1_train_n, (self.n_traindata, self.h_post, self.dim_p))

        self.x_test = np.reshape(self.x_test, (self.n_testdata, self.h_prev, self.dim_p))
        self.y0_test = np.reshape(self.y0_test, (self.n_testdata, self.h_post, self.dim_p))
        self.y1_test = np.reshape(self.y1_test, (self.n_testdata, self.h_post, self.dim_p))
        self.x_test_n = np.reshape(self.x_test_n, (self.n_testdata, self.h_prev, self.dim_p))
        self.y0_test_n = np.reshape(self.y0_test_n, (self.n_testdata, self.h_post, self.dim_p))
        self.y1_test_n = np.reshape(self.y1_test_n, (self.n_testdata, self.h_post, self.dim_p))

        # Training size
        self.n_batch_train = math.floor(self.n_traindata / self.batch_size)
        self.n_batch_test = math.floor(self.n_testdata / self.batch_size)

    def get_batch(self, idx, is_train=True):
        if is_train:
            n_batchs = self.n_batch_train
        else:
            n_batchs = self.n_batch_test

        assert idx >= 0, "idx must be non negative"
        assert idx < n_batchs, "idx must be less than the number of batches:"
        start_idx = idx * self.batch_size
        indexes_sel = range(start_idx, start_idx + self.batch_size)

        return self.get_batch_from_indexes(indexes_sel, is_train=is_train)

    def get_random_batch(self, is_train=True):
        if is_train:
            n_batchs = self.n_batch_train
        else:
            n_batchs = self.n_batch_test

        indexes_sel = np.random.permutation(n_batchs)[0:self.batch_size]

        return self.get_batch_from_indexes(indexes_sel, is_train=is_train)

    def get_batch_from_indexes(self, indexes, is_train=True):
        if is_train:
            x_data, f_data = self.x_train, self.f_train
            y0_data, y1_data, r_data = self.y0_train, self.y1_train, self.r_train
            x_data_n, f_data_n = self.x_train_n, self.f_train_n
            y0_data_n, y1_data_n, r_data_n = self.y0_train_n, self.y1_train_n, self.r_train_n
            c_data, c_data_n = self.c_train, self.c_train_n
        else:
            x_data, f_data = self.x_test, self.f_test
            y0_data, y1_data, r_data = self.y0_test, self.y1_test, self.r_test
            x_data_n, f_data_n = self.x_test_n, self.f_test_n
            y0_data_n, y1_data_n, r_data_n = self.y0_test_n, self.y1_test_n, self.r_test_n
            c_data, c_data_n = self.c_test, self.c_test_n

        x_batch = x_data[indexes, :, :]
        y0_batch = y0_data[indexes, :, :]
        y1_batch = y1_data[indexes, :, :]
        x_batch_n = x_data_n[indexes, :, :]
        y0_batch_n = y0_data_n[indexes, :, :]
        y1_batch_n = y1_data_n[indexes, :, :]

        # Feature
        if len(self.idx_f_use) > 0:
            f_batch = f_data[indexes, :]
            f_batch_n = f_data_n[indexes, :]
        else:
            f_batch, f_batch_n = [], []

        # Robustness (STL)
        if len(self.idx_r_use) > 0:
            r_batch = r_data[indexes, :]
            r_batch_n = r_data_n[indexes, :]
        else:
            r_batch, r_batch_n = [], []

        c_batch = c_data[indexes, :]
        c_batch_n = c_data_n[indexes, :]

        if not self.load_multi:
            dict_out = {'x_batch': x_batch, 'y0_batch': y0_batch, 'y1_batch': y1_batch,
                        'f_batch': f_batch, 'r_batch': r_batch, 'c_batch': c_batch,
                        'x_batch_n': x_batch_n, 'y0_batch_n': y0_batch_n, 'y1_batch_n': y1_batch_n,
                        'f_batch_n': f_batch_n, 'r_batch_n': r_batch_n, 'c_batch_n': c_batch_n}
        else:
            if is_train:
                xnear_data = self.xnear_train
                y0near_data, y1near_data = self.y0near_train, self.y1near_train
                fnear_data, rnear_data = self.fnear_train, self.rnear_train

                xnear_data_n = self.xnear_train_n
                y0near_data_n, y1near_data_n = self.y0near_train_n, self.y1near_train_n
                fnear_data_n, rnear_data_n = self.fnear_train_n, self.rnear_train_n
            else:
                xnear_data = self.xnear_test
                y0near_data, y1near_data = self.y0near_test, self.y1near_test
                fnear_data, rnear_data = self.fnear_test, self.rnear_test

                xnear_data_n = self.xnear_test_n
                y0near_data_n, y1near_data_n = self.y0near_test_n, self.y1near_test_n
                fnear_data_n, rnear_data_n = self.fnear_test_n, self.rnear_test_n

            xnear_batch = xnear_data[indexes, :, :, :]
            y0near_batch = y0near_data[indexes, :, :, :]
            y1near_batch = y1near_data[indexes, :, :, :]

            xnear_batch_n = xnear_data_n[indexes, :, :, :]
            y0near_batch_n = y0near_data_n[indexes, :, :, :]
            y1near_batch_n = y1near_data_n[indexes, :, :, :]

            if len(self.idx_f_use) > 0:  # Feature
                fnear_batch = fnear_data[indexes, :, :]
                fnear_batch_n = fnear_data_n[indexes, :, :]
            else:
                fnear_batch, fnear_batch_n = [], []

            if len(self.idx_r_use) > 0:  # Robustness (STL)
                rnear_batch = rnear_data[indexes, :, :]
                rnear_batch_n = rnear_data_n[indexes, :, :]
            else:
                rnear_batch, rnear_batch_n = [], []

            dict_out = {'x_batch': x_batch, 'y0_batch': y0_batch, 'y1_batch': y1_batch,
                        'f_batch': f_batch, 'r_batch': r_batch, 'c_batch': c_batch,
                        'x_batch_n': x_batch_n, 'y0_batch_n': y0_batch_n, 'y1_batch_n': y1_batch_n,
                        'f_batch_n': f_batch_n, 'r_batch_n': r_batch_n, 'c_batch_n': c_batch_n,
                        'xnear_batch': xnear_batch, 'y0near_batch': y0near_batch, 'y1near_batch': y1near_batch,
                        'xnear_batch_n': xnear_batch_n, 'y0near_batch_n': y0near_batch_n,
                        'y1near_batch_n': y1near_batch_n,
                        'fnear_batch': fnear_batch, 'fnear_batch_n': fnear_batch_n,
                        'rnear_batch': rnear_batch, 'rnear_batch_n': rnear_batch_n}

        return dict_out

    def shuffle_traindata(self):
        idx_random = np.random.permutation(self.n_traindata)
        self.x_train = self.x_train[idx_random, :, :]
        self.y0_train = self.y0_train[idx_random, :, :]
        self.y1_train = self.y1_train[idx_random, :, :]

        self.x_train_n = self.x_train_n[idx_random, :, :]
        self.y0_train_n = self.y0_train_n[idx_random, :, :]
        self.y1_train_n = self.y1_train_n[idx_random, :, :]

        if len(self.idx_f_use) > 0:
            self.f_train = self.f_train[idx_random, :]
            self.f_train_n = self.f_train_n[idx_random, :]

        if len(self.idx_r_use) > 0:
            self.r_train = self.r_train[idx_random, :]
            self.r_train_n = self.r_train_n[idx_random, :]

        self.c_train = self.c_train[idx_random, :]
        self.c_train_n = self.c_train_n[idx_random, :]

        if self.load_multi:
            self.xnear_train = self.xnear_train[idx_random, :, :, :]
            self.y0near_train = self.y0near_train[idx_random, :, :, :]
            self.y1near_train = self.y1near_train[idx_random, :, :, :]
            self.xnear_train_n = self.xnear_train_n[idx_random, :, :, :]
            self.y0near_train_n = self.y0near_train_n[idx_random, :, :, :]
            self.y1near_train_n = self.y1near_train_n[idx_random, :, :, :]

            if len(self.idx_f_use) > 0:
                self.fnear_train = self.fnear_train[idx_random, :, :]
                self.fnear_train_n = self.fnear_train_n[idx_random, :, :]

            if len(self.idx_r_use) > 0:
                self.rnear_train = self.rnear_train[idx_random, :, :]
                self.rnear_train_n = self.rnear_train_n[idx_random, :, :]
