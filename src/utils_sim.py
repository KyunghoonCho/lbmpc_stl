# UTILITY-FUNCTIONS (SIMULATOR)

from __future__ import print_function

import math
import numpy as np

from src.utils import *

__all__ = ["select_data_t", "select_data_t_id", "select_data_t_seglane",  # Select vehicle data
           "select_data_trange", "select_data_id_trange",
           "select_data_ids", "select_data_near",
           "get_vehicle_traj", "get_vehicle_traj_near", "get_vehicle_traj_per_id",  # Get trajectory
           "get_vehicle_vtraj_per_id", "update_heading_traj", "get_vehicle_size",
           "check_collision", "check_collision_t", "check_collision_pnts",  # Check collision
           "get_mid_pnts_lr", "get_mid_pnts_lr_dist",  # Get feature
           "get_index_seglane", "get_index_seglane_outside",
           "get_lane_cp_angle", "get_lane_cp_wrt_mtrack", "get_lane_rad_wrt_mtrack", "get_feature",
           "get_feature_sub1", "get_feature_sub2", "get_discretized_f", "get_dist2goal",
           "encode_traj", "encode_traj_2", "decode_traj", "decode_traj_2",  # Encode & Decode trajectory
           "set_initstate", "set_initstate_control", "set_initstate_pred", "get_info_t", "get_multi_info_t",  # Utils for test
           "recover_traj_pred", "check_reachgoal", "set_data_test", "get_costmap_from_trajs"]


# SELECT VEHICLE DATA -------------------------------------------------------------------------------------------------#
def select_data_t(data_v, t_cur, keep_1dim=True):
    """ Selects vehicle data w.r.t. time.
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t_cur: current time (float)
    :param keep_1dim: whether to keep output size dimension as 1 (boolean)
    """
    if len(data_v) > 0:
        idx_t_found_ = np.where(data_v[:, 0] == t_cur)
        idx_t = idx_t_found_[0]
        if len(idx_t) > 0:
            data_v_t = data_v[idx_t, :]
            if len(idx_t) == 1 and keep_1dim:
                data_v_t = data_v_t.reshape(-1)
        else:
            data_v_t, idx_t = [], []
    else:
        data_v_t, idx_t = [], []

    return data_v_t, idx_t


def select_data_t_id(data_v, t_cur, id):
    """ Selects vehicle data w.r.t. (time + vehicle-id).
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t_cur: current time (float)
    :param id: vehicle-id (int)
    """
    data_v_t_id, idx_t_id = [], []
    data_v_t, idx_t = select_data_t(data_v, t_cur)
    if len(data_v_t) > 0:
        idx_t_id = np.where(data_v_t[:, -1] == id)
        idx_t_id = idx_t_id[0]
        if len(idx_t_id) > 0:
            data_v_t_id = data_v_t[idx_t_id, :]
            idx_t_id = idx_t[idx_t_id]

    return data_v_t_id, idx_t_id


def select_data_t_seglane(data_v, t_cur, seg, lane):
    """ Selects vehicle data w.r.t. (time + seg & lane indexes).
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t_cur: current time (float)
    :param seg: indexes of seg (list)
    :param lane: indexes of lane (list)
    """
    idx_t = np.where(data_v[:, 0] == t_cur)
    idx_t = idx_t[0]
    if len(idx_t) > 0:
        data_v_in_init = data_v[idx_t, :]

        mask_t_s = np.isin(data_v_in_init[:, 7], seg)
        idx_t_s = np.nonzero(mask_t_s)
        mask_t_l = np.isin(data_v_in_init[:, 8], lane)
        idx_t_l = np.nonzero(mask_t_l)
        idx_t_sl = np.intersect1d(idx_t_s, idx_t_l)
        idx_t_sl = idx_t[idx_t_sl]
        data_v_sl = data_v[idx_t_sl, :]
    else:
        data_v_sl, idx_t_sl = [], []

    return data_v_sl, idx_t_sl


def select_data_trange(data_v, t_min, t_max):
    """ Selects vehicle data w.r.t. time-range.
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t_min: min-time (float)
    :param t_max: max-time (float)
    """
    data_v = make_numpy_array(data_v, keep_1dim=False)

    idx_found_t1 = np.where(t_min <= data_v[:, 0])
    idx_found_t2 = np.where(data_v[:, 0] <= t_max)
    idx_found_t = np.intersect1d(idx_found_t1, idx_found_t2)
    data_v_trange = data_v[idx_found_t, :]

    return data_v_trange


def select_data_id_trange(data_v, id_tv, t_min, t_max):
    """ Selects vehicle data w.r.t. (id + time-length).
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param id_tv: id of target vehicle (int)
    :param t_min: min-time (float)
    :param t_max: max-time (float)
    """
    data_v = make_numpy_array(data_v, keep_1dim=False)
    data_tv, data_ov = [], []

    idx_found_t1 = np.where(t_min <= data_v[:, 0])
    idx_found_t2 = np.where(data_v[:, 0] <= t_max)
    idx_found_t = np.intersect1d(idx_found_t1, idx_found_t2)
    if len(idx_found_t) > 0:
        data_v = data_v[idx_found_t, :]

        idx_tv_ = np.where(data_v[:, -1] == id_tv)
        idx_tv = idx_tv_[0]
        if len(idx_tv) > 0:
            data_tv = data_v[idx_tv, :]

        idx_ov = np.setdiff1d(np.arange(0, data_v.shape[0]), idx_tv)
        if len(idx_ov) > 0:
            data_ov = data_v[idx_ov, :]

    return data_v, data_tv, data_ov


def select_data_ids(data_v, ids):
    """ Selects vehicle data w.r.t. ids.
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param ids: vehicle ids (list, ndarray)
    """
    if len(data_v) > 0:
        data_v = make_numpy_array(data_v, keep_1dim=False)

        if ~isinstance(ids, np.ndarray):
            ids = np.array(ids, dtype=np.int32)

        dim_data = data_v.shape[1]

        id_sel_c = []
        if len(ids) > 0:
            idx_sel_ = np.where(ids != -1)
            idx_sel = idx_sel_[0]
            if len(idx_sel) > 0:
                id_sel_c = ids[idx_sel]

        data_v_sel = np.zeros((data_v.shape[0], dim_data), dtype=np.float32)
        idx_v_sel = np.zeros((data_v.shape[0], ), dtype=np.int32)
        cnt_data_v_sel = 0
        if len(id_sel_c) > 0:
            id_data_tmp = data_v[:, -1]
            id_data_tmp = id_data_tmp.astype(dtype=np.int32)

            for nidx_d in range(0, id_sel_c.shape[0]):
                id_sel_c_sel = id_sel_c[nidx_d]
                idx_sel_2_ = np.where(id_data_tmp == id_sel_c_sel)
                idx_sel_2 = idx_sel_2_[0]

                if len(idx_sel_2) > 0:
                    idx_update_tmp = np.arange(cnt_data_v_sel, cnt_data_v_sel + len(idx_sel_2))
                    data_v_sel[idx_update_tmp, :] = data_v[idx_sel_2, :]
                    idx_v_sel[idx_update_tmp] = idx_sel_2
                    cnt_data_v_sel = cnt_data_v_sel + len(idx_sel_2)

        if cnt_data_v_sel == 0:
            data_v_sel, data_v_rest = [], np.copy(data_v)
        else:
            data_v_sel = data_v_sel[np.arange(0, cnt_data_v_sel), :]
            idx_v_sel = idx_v_sel[np.arange(0, cnt_data_v_sel)]
            idx_v_rest = np.setdiff1d(np.arange(0, data_v.shape[0]), idx_v_sel)
            data_v_rest = data_v[idx_v_rest, :]
    else:
        data_v_sel, data_v_rest = [], []

    return data_v_sel, data_v_rest


def select_data_near(data_v, id_near):
    """ Selects (near) vehicle data.
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param id_near: near vehicle ids (list, ndarray)
    """
    data_v = make_numpy_array(data_v, keep_1dim=False)
    id_near = make_numpy_array(id_near, keep_1dim=True)

    if len(data_v) > 0:
        dim_data = data_v.shape[1]

        data_v_near = np.zeros((data_v.shape[0], dim_data), dtype=np.float32)
        cnt_data_v_near = 0
        data_v_near_list = []
        if len(id_near) > 0:
            id_data_tmp = data_v[:, -1]
            id_data_tmp = id_data_tmp.astype(dtype=np.int32)

            for nidx_d in range(0, id_near.shape[0]):
                id_near_sel = id_near[nidx_d]
                if id_near_sel == -1:
                    data_v_near_tmp = -1
                else:
                    idx_near_sel_2_ = np.where(id_data_tmp == id_near_sel)
                    idx_near_sel_2 = idx_near_sel_2_[0]

                    if len(idx_near_sel_2) > 0:
                        idx_update_tmp = np.arange(cnt_data_v_near, cnt_data_v_near + len(idx_near_sel_2))
                        data_v_near[idx_update_tmp, :] = data_v[idx_near_sel_2, :]
                        cnt_data_v_near = cnt_data_v_near + len(idx_near_sel_2)

                        data_v_near_tmp = data_v[idx_near_sel_2, :]
                    else:
                        data_v_near_tmp = -1
                data_v_near_list.append(data_v_near_tmp)

        if cnt_data_v_near == 0:
            data_v_near = []
        else:
            data_v_near = data_v_near[np.arange(0, cnt_data_v_near), :]
    else:
        data_v_near, data_v_near_list = [], []

    return data_v_near, data_v_near_list


# GET TRAJECTORY ------------------------------------------------------------------------------------------------------#
def get_vehicle_traj(data_v, t, horizon, handle_remain=0):
    """ Gets vehicle-trajectory.
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t: current time (int)
    :param horizon: horizon or length of trajectory (int)
    :param handle_remain: method to handle remain values (int)
    """
    if len(data_v) > 0:
        data_v = make_numpy_array(data_v, keep_1dim=False)

        id_unique_ = data_v[:, -1]
        id_unique = np.unique(id_unique_, axis=0)
        id_unique = id_unique.astype(dtype=np.int32)

        size_out = np.zeros((id_unique.shape[0], 2), dtype=np.float32)
        traj_out = []
        for nidx_i in range(0, id_unique.shape[0]):
            idx_sel_1_ = np.where(data_v[:, -1] == id_unique[nidx_i])
            idx_sel_1 = idx_sel_1_[0]

            data_vehicle_per_id = data_v[idx_sel_1, :]
            size_out[nidx_i, :] = [data_v[idx_sel_1[0], 6], data_v[idx_sel_1[0], 5]]

            traj_out_tmp = np.zeros((horizon + 1, 3), dtype=np.float32)
            for nidx_t in range(t, t + horizon + 1):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    traj_out_tmp[nidx_t - t, :] = data_vehicle_sel[1:4]
                else:
                    if handle_remain == 0:
                        traj_out_tmp[nidx_t - t, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out_tmp[nidx_t - t, :] = traj_out_tmp[nidx_t - t - 1, :]
                    else:
                        traj_out_tmp[nidx_t - t, :] = [traj_out_tmp[0, 0] - 1000.0, traj_out_tmp[0, 1] - 1000.0,
                                                       0.0]

            traj_out.append(traj_out_tmp)
    else:
        id_unique, traj_out, size_out = [], [], []

    return id_unique, traj_out, size_out


def get_vehicle_traj_near(data_v_list, id_near, idx_st, horizon, do_reverse=0, handle_remain=1, val_empty=-400.0):
    """ Gets vehicle-trajectory (near).
    :param data_v_list: (list)-> (ndarray) t x y theta v length width tag_segment tag_lane id (dim = N x 10)
                                  (width > length)
                              -> (scalar) -1 (empty)
                        [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    :param id_near: near vehicle id [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr] (ndarray)
    :param idx_st: start index (int)
    :param horizon: horizon or length of trajectory (int)
    :param do_reverse: whether to find reverse trajectory (boolean)
    :param handle_remain: method to handle remain values (int)
    :param val_empty: position value for empty-trajectory (float)
    """
    id_near = make_numpy_array(id_near, keep_1dim=True)

    len_list = len(data_v_list)

    traj_out, size_out = [], []

    for nidx_l in range(0, len_list):
        data_vehicle_list_sel = data_v_list[nidx_l]
        data_vehicle_list_sel = make_numpy_array(data_vehicle_list_sel, keep_1dim=False)

        id_near_sel = id_near[nidx_l]
        if id_near_sel == -1:
            size_out_tmp = np.zeros((2,), dtype=np.float32)

            traj_out_tmp = val_empty * np.ones((horizon + 1, 3), dtype=np.float32)
            traj_out_tmp[:, 2] = 0
        else:
            size_out_tmp = np.array([data_vehicle_list_sel[0, 6], data_vehicle_list_sel[0, 5]], dtype=np.float32)

            traj_out_tmp = np.zeros((horizon + 1, 3), dtype=np.float32)

            iter_start = idx_st
            if do_reverse == 0:
                iter_end = idx_st + horizon + 1
                iter_step = +1
            else:
                iter_end = idx_st - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_list_sel[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - idx_st)
                else:
                    idx_cur_tmp = nidx_t - idx_st

                if len(idx_sel_2) > 0:
                    traj_out_tmp[idx_cur_tmp, :] = data_vehicle_list_sel[idx_sel_2[0], 1:4]
                else:
                    if handle_remain == 0:
                        traj_out_tmp[idx_cur_tmp, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out_tmp[idx_cur_tmp, :] = traj_out_tmp[idx_cur_tmp - 1, :]
                    elif handle_remain == 2:
                        traj_out_tmp[idx_cur_tmp, :] = [traj_out_tmp[0, 0] - 1000.0, traj_out_tmp[0, 1] - 1000.0, 0.0]
                    else:
                        diff_tmp = traj_out_tmp[idx_cur_tmp - 1, :] - traj_out_tmp[idx_cur_tmp - 2, :]
                        traj_out_tmp[idx_cur_tmp, :] = traj_out_tmp[idx_cur_tmp - 1, :] + diff_tmp

            if do_reverse == 1:
                traj_out_tmp = np.flipud(traj_out_tmp)

        traj_out.append(traj_out_tmp)
        size_out.append(size_out_tmp)

    return traj_out, size_out


def get_vehicle_traj_per_id(data_v, idx_st, id_in, horizon, do_reverse=0, handle_remain=1):
    """ Gets vehicle-trajectory (per id).
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param idx_st: start index (int)
    :param id_in: vehicle id (int)
    :param horizon: horizon or length of trajectory (int)
    :param do_reverse: whether to find reverse trajectory (boolean)
    :param handle_remain: method to handle remain values (int)
    :return:
    """
    data_v = make_numpy_array(data_v, keep_1dim=False)

    traj_out, size_out = [], []

    if len(data_v) > 0:
        idx_sel_1_ = np.where(data_v[:, -1] == id_in)
        idx_sel_1 = idx_sel_1_[0]

        if len(idx_sel_1) > 0:
            _data_vehicle_per_id = data_v[idx_sel_1, :]
            if do_reverse:
                _data_vehicle_per_id_after = np.reshape(_data_vehicle_per_id[-1, :], (1, -1))
                _data_vehicle_per_id_after = np.tile(_data_vehicle_per_id_after, (2, 1))
                _data_vehicle_per_id_after[:, 0] = [_data_vehicle_per_id[-1, 0] + 1, _data_vehicle_per_id[-1, 0] + 2]
                data_vehicle_per_id = np.concatenate((_data_vehicle_per_id, _data_vehicle_per_id_after), axis=0)
            else:
                _data_vehicle_per_id_before = np.reshape(_data_vehicle_per_id[0, :], (1, -1))
                _data_vehicle_per_id_before = np.tile(_data_vehicle_per_id_before, (2, 1))
                _data_vehicle_per_id_before[:, 0] = [_data_vehicle_per_id[0, 0] - 2, _data_vehicle_per_id[0, 0] - 1]
                data_vehicle_per_id = np.concatenate((_data_vehicle_per_id_before, _data_vehicle_per_id), axis=0)

            size_out = [data_v[idx_sel_1[0], 6], data_v[idx_sel_1[0], 5]]

            traj_out = np.zeros((horizon + 1, 3), dtype=np.float32)

            iter_start = idx_st
            if do_reverse == 0:
                iter_end = idx_st + horizon + 1
                iter_step = +1
            else:
                iter_end = idx_st - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - idx_st)
                else:
                    idx_cur_tmp = nidx_t - idx_st

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    traj_out[idx_cur_tmp, :] = data_vehicle_sel[1:4]
                else:
                    if handle_remain == 0:
                        traj_out[idx_cur_tmp, :] = [np.nan, np.nan, np.nan]
                    elif handle_remain == 1:
                        traj_out[idx_cur_tmp, :] = traj_out[idx_cur_tmp - 1, :]
                    elif handle_remain == 2:
                        traj_out[idx_cur_tmp, :] = [traj_out[0, 0] - 1000.0, traj_out[0, 1] - 1000.0, 0.0]
                    else:
                        diff_tmp = traj_out[idx_cur_tmp - 1, :] - traj_out[idx_cur_tmp - 2, :]
                        traj_out[idx_cur_tmp, :] = traj_out[idx_cur_tmp - 1, :] + diff_tmp

            if do_reverse == 1:
                traj_out = np.flipud(traj_out)

    return traj_out, size_out


def get_vehicle_vtraj_per_id(data_v, idx_st, id_in, horizon, do_reverse=0, handle_remain=1):
    """ Gets vehicle-vtrajectory (velocity per id).
    :param data_v: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param idx_st: start index (int)
    :param id_in: vehicle id (int)
    :param horizon: horizon or length of trajectory (int)
    :param do_reverse: whether to find reverse trajectory (boolean)
    :param handle_remain: method to handle remain values (int)
    """
    data_v = make_numpy_array(data_v, keep_1dim=False)

    vtraj_out = []

    if len(data_v) > 0:
        if ~isinstance(data_v, np.ndarray):
            data_v = np.array(data_v)

        idx_sel_1_ = np.where(data_v[:, -1] == id_in)
        idx_sel_1 = idx_sel_1_[0]

        if len(idx_sel_1) > 0:
            data_vehicle_per_id = data_v[idx_sel_1, :]

            vtraj_out = np.zeros((horizon + 1, ), dtype=np.float32)

            iter_start = idx_st
            if do_reverse == 0:
                iter_end = idx_st + horizon + 1
                iter_step = +1
            else:
                iter_end = idx_st - horizon - 1
                iter_step = -1

            for nidx_t in range(iter_start, iter_end, iter_step):
                idx_sel_2_ = np.where(data_vehicle_per_id[:, 0] == nidx_t)
                idx_sel_2 = idx_sel_2_[0]

                if do_reverse == 1:
                    idx_cur_tmp = -1 * (nidx_t - idx_st)
                else:
                    idx_cur_tmp = nidx_t - idx_st

                if len(idx_sel_2) > 0:
                    data_vehicle_sel = data_vehicle_per_id[idx_sel_2[0], :]
                    vtraj_out[idx_cur_tmp] = data_vehicle_sel[4]
                else:
                    if handle_remain == 0:
                        vtraj_out[nidx_t - idx_st] = np.nan
                    elif handle_remain == 1:
                        vtraj_out[idx_cur_tmp] = vtraj_out[idx_cur_tmp - 1]
                    else:
                        vtraj_out[nidx_t - idx_st, :] = 0

            if do_reverse == 1:
                vtraj_out = np.flipud(vtraj_out)

    return vtraj_out


def update_heading_traj(traj, theta_thres=0.314):
    """ Updates heading of trajectory.
    :param traj: trajectory [x, y, heading] (dim = N x 3)
    :param theta_thres: theta threshold """

    traj = make_numpy_array(traj, keep_1dim=False)
    len_traj = traj.shape[0]
    traj_new = np.copy(traj)
    if len_traj > 1:
        for nidx_d in range(len_traj - 1):
            nidx_d_next = min(nidx_d + 4, len_traj - 1)
            p_0 = traj[nidx_d, :]
            p_1 = traj[nidx_d_next, :]

            p_0 = np.reshape(p_0, -1)
            p_1 = np.reshape(p_1, -1)
            diff_xy = p_1[0:2] - p_0[0:2]

            theta_0 = p_1[2]
            theta_1 = math.atan2(diff_xy[1], diff_xy[0])
            diff_theta = get_diff_angles(theta_1, theta_0)

            if abs(diff_theta) > theta_thres:
                theta_delta = np.sign(diff_theta) * theta_thres
            else:
                theta_delta = diff_theta

            theta_new_ = theta_0 + theta_delta * 0.5
            theta_new = angle_handle(theta_new_)
            traj_new[nidx_d + 1, 2] = theta_new

    return traj_new


def get_vehicle_size(data_v, id):
    """ Gets vehicle-size (id).
    :param data_v: t x y theta v length width tag_segment tag_lane id (width > length) (dim = N x 10)
    :param id: vehicle id
    """
    id = make_numpy_array(id, keep_1dim=True)
    size_out = []

    for nidx_l in range(0, len(id)):
        id_sel = id[nidx_l]
        if id_sel == -1:
            size_out_tmp = np.zeros((2,), dtype=np.float32)
        else:
            data_tv, _ = select_data_ids(data_v, [id_sel])
            data_tv = make_numpy_array(data_tv, keep_1dim=False)
            size_out_tmp = np.array([data_tv[0, 6], data_tv[0, 5]], dtype=np.float32)

        size_out.append(size_out_tmp)

    return size_out


def check_collision(data_tv, data_ov):
    """ Checks collision.
    :param data_tv: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = 10, width > length)
    :param data_ov: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    """
    data_tv = make_numpy_array(data_tv, keep_1dim=True)
    data_tv_r = np.reshape(data_tv[1:3], (1, 2))

    data_ov = make_numpy_array(data_ov, keep_1dim=False)
    num_ov = data_ov.shape[0]

    # Near distance threshold (30, -1)
    if num_ov >= 10:
        dist_r = 12.5
    else:
        dist_r = -1

    if dist_r > 0:
        # Get near distance other vehicle data
        diff_array = np.repeat(data_tv_r, data_ov.shape[0], axis=0) - data_ov[:, 1:3]
        dist_array = np.sqrt(np.sum(diff_array * diff_array, axis=1))

        idx_sel_ = np.where(dist_array <= dist_r)
        idx_sel = idx_sel_[0]
        data_oc_sel = data_ov[idx_sel, :]
    else:
        data_oc_sel = data_ov

    if len(data_oc_sel.shape) == 1:
        data_oc_sel = np.reshape(data_oc_sel, (1, -1))

    # Get pnts of box (target vehicle)
    # pnts_out_ = get_box_pnts(data_t[1], data_t[2], data_t[3], data_t[6], data_t[5])

    nx_col = max(int(data_tv[6]/0.15), 20)
    ny_col = max(int(data_tv[5]/0.15), 10)
    pnts_out_ = get_box_pnts_precise(data_tv[1], data_tv[2], data_tv[3], data_tv[6], data_tv[5], nx=nx_col, ny=ny_col)
    pnts_m_ = get_m_pnts(data_tv[1], data_tv[2], data_tv[3], data_tv[6], nx=nx_col)
    pnts_out = np.concatenate((pnts_out_, pnts_m_), axis=0)

    len_oc = data_oc_sel.shape[0]
    is_collision = False
    for nidx_d1 in range(0, len_oc):  # For all other vehicles
        # Get pnts of box (other vehicle)
        data_oc_sel_tmp = data_oc_sel[nidx_d1, :]
        pnts_oc_out = get_box_pnts(data_oc_sel_tmp[1], data_oc_sel_tmp[2], data_oc_sel_tmp[3],
                                   data_oc_sel_tmp[6], data_oc_sel_tmp[5])

        is_out = 0
        for nidx_d2 in range(0, pnts_out.shape[0]):  # For all pnts (target vehicle)
            pnts_out_sel = pnts_out[nidx_d2, :]
            is_in = inpolygon(pnts_out_sel[0], pnts_out_sel[1], pnts_oc_out[:, 0], pnts_oc_out[:, 1])
            if is_in == 1:
                is_out = 1
                break

        if is_out == 1:
            is_collision = True
            break

    return is_collision


def check_collision_t(data_ev, data_ov, t_cur):
    """ Checks collision w.r.t. time.
    :param data_ev: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = 10, width > length)
    :param data_ov: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    :param t_cur: current time (float)
    """
    if len(data_ov) > 0:
        data_ev = make_numpy_array(data_ev, keep_1dim=True)
        data_ov = make_numpy_array(data_ov, keep_1dim=False)

        idx_sel_ = np.where(data_ov[:, 0] == t_cur)
        idx_sel = idx_sel_[0]
        if len(idx_sel) > 0:
            data_ov_cur = data_ov[idx_sel, :]
            is_collision = check_collision(data_ev, data_ov_cur)
        else:
            is_collision = False
    else:
        is_collision = False

    return is_collision


def check_collision_pnts(pnts, data_ov):
    """ Checks collision (points).
    :param pnts: points (x, y) (dim = N x 2)
    :param data_ov: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
    """
    len_ov = data_ov.shape[0]
    is_collision = False
    for nidx_d1 in range(0, len_ov):  # For all other vehicles
        # Get pnts of box (other vehicle)
        data_ov_sel_tmp = data_ov[nidx_d1, :]
        pnts_ov_out = get_box_pnts(data_ov_sel_tmp[1], data_ov_sel_tmp[2], data_ov_sel_tmp[3], data_ov_sel_tmp[6],
                                   data_ov_sel_tmp[5])

        is_out = 0
        for nidx_d2 in range(0, pnts.shape[0]):  # For all pnts (target vehicle)
            pnts_sel = pnts[nidx_d2, :]
            is_in = inpolygon(pnts_sel[0], pnts_sel[1], pnts_ov_out[:, 0], pnts_ov_out[:, 1])
            if is_in == 1:
                is_out = 1
                break

        if is_out == 1:
            is_collision = True
            break
    return is_collision

# GET FEATURE ---------------------------------------------------------------------------------------------------------#
def get_mid_pnts_lr(pnts_l, pnts_r, num_intp):
    """ Gets middle points from two points (track: left-right).
    :param pnts_l: left side points (ndarray, dim = N x 2)
    :param pnts_r: right side points (ndarray, dim = N x 2)
    :param num_intp: interpolation number (int)
    """
    pnts_l = make_numpy_array(pnts_l, keep_1dim=False)
    pnts_r = make_numpy_array(pnts_r, keep_1dim=False)

    pnt_x_l_tmp, pnt_y_l_tmp = pnts_l[:, 0].reshape(-1), pnts_l[:, 1].reshape(-1)
    pnt_x_r_tmp, pnt_y_r_tmp = pnts_r[:, 0].reshape(-1), pnts_r[:, 1].reshape(-1)
    pnt_t_tmp = np.arange(0, pnt_x_r_tmp.shape[0])

    pnt_t_range = np.linspace(min(pnt_t_tmp), max(pnt_t_tmp), num=num_intp)

    x_l_intp = np.interp(pnt_t_range, pnt_t_tmp, pnt_x_l_tmp)
    x_r_intp = np.interp(pnt_t_range, pnt_t_tmp, pnt_x_r_tmp)
    y_l_intp = np.interp(pnt_t_range, pnt_t_tmp, pnt_y_l_tmp)
    y_r_intp = np.interp(pnt_t_range, pnt_t_tmp, pnt_y_r_tmp)

    pnts_l_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_r_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_l_intp[:, 0], pnts_l_intp[:, 1] = x_l_intp, y_l_intp
    pnts_r_intp[:, 0], pnts_r_intp[:, 1] = x_r_intp, y_r_intp
    pnts_c_intp = (pnts_l_intp + pnts_r_intp) / 2

    return pnts_c_intp


def get_mid_pnts_lr_dist(pnts_l, pnts_r, dist_intv):
    """ Gets middle points from two points w.r.t. dist_intv (track: left-right).
    :param pnts_l: left side points (ndarray, dim = N x 2)
    :param pnts_r: right side points (ndarray, dim = N x 2)
    :param dist_intv: interval distance (float)
    """
    pnts_l_in = make_numpy_array(pnts_l, keep_1dim=False)
    len_pnts_l_in = pnts_l_in.shape[0]
    diff_tmp = pnts_l_in[np.arange(0, len_pnts_l_in - 1), 0:2] - pnts_l_in[np.arange(1, len_pnts_l_in), 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    dist_sum = np.sum(dist_tmp)
    num_intp_l = int(dist_sum / dist_intv)

    pnts_r_in = make_numpy_array(pnts_r, keep_1dim=False)
    len_pnts_r_in = pnts_r_in.shape[0]
    diff_tmp = pnts_r_in[np.arange(0, len_pnts_r_in - 1), 0:2] - pnts_r_in[np.arange(1, len_pnts_r_in), 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    dist_sum = np.sum(dist_tmp)
    num_intp_r = int(dist_sum / dist_intv)

    num_intp = int((num_intp_l + num_intp_r) / 2.0)

    pnts_c_intp = get_mid_pnts_lr(pnts_l, pnts_r, num_intp)

    return pnts_c_intp


def get_index_seglane(pos_i, pnts_poly_track):
    """ Gets indexes of segment and lane.
    :param pos_i: point (dim = 2)
    :param pnts_poly_track: track-points
    :return: curseg, curlane
    """
    pos_i = make_numpy_array(pos_i, keep_1dim=True)

    # Check on which track this car is
    num_check = 3
    curseg = -1 * np.ones((num_check, ), dtype=np.int32)
    curlane = -1 * np.ones((num_check, ), dtype=np.int32)
    cnt_check = 0
    for segidx in range(0, len(pnts_poly_track)):
        pnts_poly_seg = pnts_poly_track[segidx]

        for laneidx in range(0, len(pnts_poly_seg)):
            pnts_poly_lane = pnts_poly_seg[laneidx]

            point_is_in_hull = inpolygon(pos_i[0], pos_i[1], pnts_poly_lane[:, 0], pnts_poly_lane[:, 1])
            if point_is_in_hull:
                cnt_check = cnt_check + 1
                curseg[cnt_check - 1] = int(segidx)
                curlane[cnt_check - 1] = int(laneidx)

                if cnt_check >= num_check:
                    break
    if cnt_check == 0:
        curseg, curlane = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
    else:
        curseg, curlane = curseg[0:cnt_check], curlane[0:cnt_check]

    return curseg, curlane


def get_index_seglane_outside(pos_i, pnts_poly_track):
    """ Gets indexes of segment and lane when point is outside of track.
    :param pos_i: point (dim = 2)
    :param pnts_poly_track: track-points
    :return: curseg, curlane
    """
    pos_i = make_numpy_array(pos_i, keep_1dim=True)

    # Check on which track this car is
    curseg, curlane = np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32)
    dist_cur = 100000
    for segidx in range(0, len(pnts_poly_track)):
        pnts_poly_seg = pnts_poly_track[segidx]

        for laneidx in range(0, len(pnts_poly_seg)):
            pnts_poly_lane = pnts_poly_seg[laneidx]

            pnt_mean = np.mean(pnts_poly_lane[:, 0:2], axis=0)
            pnt_mean = pnt_mean.reshape(-1)

            vec_i2mean = np.array([pos_i[0] - pnt_mean[0], pos_i[1] - pnt_mean[1]], dtype=np.float32)
            dist_i2mean = norm(vec_i2mean)

            if dist_i2mean < dist_cur:
                dist_cur = dist_i2mean
                curseg[0] = segidx
                curlane[0] = laneidx

    return curseg, curlane


def get_lane_cp_angle(pnt, pnts_poly_track, pnts_lr_border_track):
    """ Gets lane angle.
    :param pnt: in point (dim = 2)
    :param pnts_poly_track: track-points (polygon)
    :param pnts_lr_border_track: track-points (left-right)
    :return: pnt_center, rad_center
    """
    pnt = make_numpy_array(pnt, keep_1dim=True)

    curseg_, curlane_ = get_index_seglane(pnt[0:2], pnts_poly_track)
    curseg, curlane = curseg_[0], curlane_[0]
    if curseg == -1 or curlane == -1:
        curseg_, curlane_ = get_index_seglane_outside(pnt[0:2], pnts_poly_track)
        curseg, curlane = curseg_[0], curlane_[0]

    if curseg == -1 or curlane == -1:
        pnt_center = np.copy(pnt)
        rad_center = 0.0
    else:
        pnts_lr_border_lane = pnts_lr_border_track[curseg][curlane]
        pnts_left = pnts_lr_border_lane[0]
        pnts_right = pnts_lr_border_lane[1]
        pnt_minleft, _ = get_closest_pnt(pnt[0:2], pnts_left)
        pnt_minright, _ = get_closest_pnt(pnt[0:2], pnts_right)
        # pnt_center = (pnt_minleft + pnt_minright) / 2
        vec_l2r = pnt_minright[0:2] - pnt_minleft[0:2]
        rad = math.atan2(vec_l2r[1], vec_l2r[0])
        rad_center = (rad + math.pi / 2)
        pnt_center = (pnt_minleft + pnt_minright) / 2

    return pnt_center, rad_center


def get_lane_cp_wrt_mtrack(sim_track, pos, seg, lane):
    """ Gets lane-cp w.r.t. mtrack.
    :param sim_track: track-info
    :param pos: x, y (dim = 2)
    :param seg: segment index
    :param lane: lane index
    :return: pnt_c_out
    """
    # Find indexes of seg & lane
    if seg == -1 or lane == -1:
        seg_, lane_ = get_index_seglane(pos, sim_track.pnts_poly_track)
        seg, lane = seg_[0], lane_[0]

    pnts_c_tmp = sim_track.pnts_m_track[seg][lane]  # [0, :] start --> [end, :] end

    # Find middle points
    if seg < (sim_track.num_seg - 1):
        seg_next = seg + 1
    else:
        seg_next = 0 if sim_track.is_circular else -1

    if seg_next > -1:
        child_tmp = sim_track.idx_child[seg][lane]
        for nidx_tmp in range(0, len(child_tmp)):
            pnts_c_next_tmp = sim_track.pnts_m_track[seg_next][sim_track.idx_child[seg][lane][nidx_tmp]]
            # [0, :] start --> [end, :] end
            pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
        pnts_c = pnts_c_tmp
    else:
        pnts_c = pnts_c_tmp

    # Find lane cp w.r.t. middle points
    pnt_c_out, dist_c_out = get_closest_pnt_intp(pos[0:2], pnts_c, num_intp=100)

    return pnt_c_out


def get_lane_rad_wrt_mtrack(sim_track, pos, seg, lane, delta=5):
    """ Gets lane-angle w.r.t. mtrack.
    :param sim_track: track-info
    :param pos: x, y (dim = 2)
    :param seg: segment index (int)
    :param lane: lane index (int)
    :param delta: time-step (int)
    :return: pnt_c_out
    """
    # Find indexes of seg & lane
    if seg == -1 or lane == -1:
        seg_, lane_ = get_index_seglane(pos, sim_track.pnts_poly_track)
        seg, lane = seg_[0], lane_[0]

    pnts_c_tmp = sim_track.pnts_m_track[seg][lane]  # [0, :] start --> [end, :] end

    # Find middle points
    if seg < (sim_track.num_seg - 1):
        child_tmp = sim_track.idx_child[seg][lane]
        for nidx_tmp in range(0, len(child_tmp)):
            pnts_c_next_tmp = sim_track.pnts_m_track[seg + 1][sim_track.idx_child[seg][lane][nidx_tmp]]
            # [0, :] start --> [end, :] end
            pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
        pnts_c = pnts_c_tmp
    else:
        pnts_c = pnts_c_tmp

    # Find lane angle w.r.t. middle points
    len_pnts_c = pnts_c.shape[0]
    pos_r = np.reshape(pos, (1, 2))
    diff_tmp = np.tile(pos_r, (len_pnts_c, 1)) - pnts_c[:, 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    idx_cur = np.argmin(dist_tmp, axis=0)

    pnt_c_cur = pnts_c[idx_cur, 0:2]
    if (idx_cur + delta) <= (len_pnts_c - 1):
        idx_next = idx_cur + delta
    else:
        idx_next = len_pnts_c - 1

    if (idx_cur - delta) >= 0:
        idx_prev = idx_cur - delta
    else:
        idx_prev = 0

    if idx_cur == 0:
        pnt_c_next = pnts_c[idx_next, 0:2]
        angle_c = math.atan2(pnt_c_next[1] - pnt_c_cur[1], pnt_c_next[0] - pnt_c_cur[0])
    elif idx_cur == pnts_c.shape[0]:
        pnt_c_prev = pnts_c[idx_prev, 0:2]
        angle_c = math.atan2(pnt_c_cur[1] - pnt_c_prev[1], pnt_c_cur[0] - pnt_c_prev[0])
    else:
        pnt_c_next = pnts_c[idx_next, 0:2]
        pnt_c_prev = pnts_c[idx_prev, 0:2]
        angle_c_f = math.atan2(pnt_c_next[1] - pnt_c_cur[1], pnt_c_next[0] - pnt_c_cur[0])
        angle_c_b = math.atan2(pnt_c_cur[1] - pnt_c_prev[1], pnt_c_cur[0] - pnt_c_prev[0])

        # if abs(angle_c_f) < 0.01:
        #     angle_c = angle_c_b
        # elif abs(angle_c_b) < 0.01:
        #     angle_c = angle_c_f
        # else:
        #     angle_c = (angle_c_f + angle_c_b) / 2.0

        if abs(angle_c_f) < 0.01:
            angle_c = angle_c_b
        else:
            angle_c = angle_c_f

    return angle_c


def get_feature(sim_track, data_tv, data_ov, use_intp=0, do_precise=True):
    """ Gets feature.
    :param sim_track: track-info
    :param data_tv: target(ego) vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = 10)
    :param data_ov: other vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10)
    :param use_intp: whether to use interpolation (pnt_left, pnt_right, pnt_center, lane_angle)
    :param do_precise: whether to compute precise features (ov_ld_dist_array)
    :return: f_out, id_near, pnts_feature_ev, pnts_feature_ov, rad_center, d_cf
    """
    data_tv = make_numpy_array(data_tv, keep_1dim=True)
    if len(data_ov) > 0:
        data_ov = make_numpy_array(data_ov, keep_1dim=False)

    # Get car-info
    pos_i = data_tv[1:4]  # x, y, theta(rad)
    carlen = data_tv[6]
    carwidth = data_tv[5]

    if len(data_ov) > 0:
        num_car = data_ov.shape[0]
    else:
        num_car = 0

    # Check on which track this car is
    if int(data_tv[7]) == -1 or int(data_tv[8]) == -1:
        curseg, curlane = get_index_seglane(pos_i, sim_track.pnts_poly_track)
        data_tv[7], data_tv[8] = curseg[0], curlane[0]

    # FIND FEATURE
    # A. GET LANE DEVIATION DIST
    curseg, curlane, lanedev_rad, lanedev_dist, lanedev_rdist, lanewidth, pnt_center, rad_center, pnt_left, pnt_right \
        = get_feature_sub1(sim_track, data_tv, use_intp)
    seg_i, lane_i = curseg[0], curlane[0]

    lanedev_dist_scaled = lanedev_dist / lanewidth
    lanedev_rdist_scaled = lanedev_rdist / lanewidth

    # B. FRONTAL AND REARWARD DISTANCES (LEFT, CENTER, RIGHT)
    max_dist = 40  # maximum-distance
    id_lf, id_lr, id_rf, id_rr, id_cf, id_cr, idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr, d_lf, d_lr, d_rf, d_rr, \
    d_cf, d_cr, pnt_lf, pnt_lr, pnt_rf, pnt_rr, pnt_cf, pnt_cr = get_feature_sub2(sim_track, pos_i, seg_i, lane_i,
                                                                                  data_ov, num_car, max_dist)

    # Set lr-dist
    idx_array = np.array([idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr], dtype=np.int32)
    ov_ld_dist_array = np.zeros((6,), dtype=np.float32)
    if do_precise:
        for nidx_sr in range(0, idx_array.shape[0]):
            idx_tmp = idx_array[nidx_sr]
            if idx_tmp >= 0:
                _, _, _, ld_dist_tmp, ld_rdist_tmp, lw_tmp, _, _, _, _ = \
                    get_feature_sub1(sim_track, data_ov[idx_tmp, :], use_intp=use_intp)
                ov_ld_dist_array[nidx_sr] = ld_dist_tmp / lw_tmp

    d_lf_scaled, d_lr_scaled = d_lf / lanewidth, d_lr / lanewidth
    d_cf_scaled, d_cr_scaled = d_cf / lanewidth, d_cr / lanewidth
    d_rf_scaled, d_rr_scaled = d_rf / lanewidth, d_rr / lanewidth

    f_out = np.array([lanedev_rad, lanedev_dist, lanedev_rdist, lanewidth, lanedev_dist_scaled, lanedev_rdist_scaled,
                      d_lf_scaled, d_lr_scaled, d_cf_scaled, d_cr_scaled, d_rf_scaled, d_rr_scaled,
                      ov_ld_dist_array[0], ov_ld_dist_array[1], ov_ld_dist_array[2], ov_ld_dist_array[3],
                      ov_ld_dist_array[4], ov_ld_dist_array[5], data_tv[4], carlen, carwidth], dtype=np.float32)

    id_lr = -1 if id_lf == id_lr else id_lr
    id_rr = -1 if id_rf == id_rr else id_rr
    id_cr = -1 if id_cf == id_cr else id_cr

    id_near = np.array([id_lf, id_lr, id_rf, id_rr, id_cf, id_cr], dtype=np.int32)

    pnts_feature_ev = np.zeros((3, 2), dtype=np.float32)
    pnts_feature_ev[0, :] = pnt_center
    pnts_feature_ev[1, :] = pnt_left
    pnts_feature_ev[2, :] = pnt_right

    pnts_feature_ov = np.zeros((6, 2), dtype=np.float32)
    pnts_feature_ov[0, :] = pnt_lf
    pnts_feature_ov[1, :] = pnt_lr
    pnts_feature_ov[2, :] = pnt_cf
    pnts_feature_ov[3, :] = pnt_cr
    pnts_feature_ov[4, :] = pnt_rf
    pnts_feature_ov[5, :] = pnt_rr

    return f_out, id_near, pnts_feature_ev, pnts_feature_ov, rad_center, d_cf


def get_feature_sub1(sim_track, data_t, use_intp=0):
    """ Gets feature (sub1).
    :param sim_track: track-info
    :param data_t: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = 10)
    :param use_intp: whether to use interpolation (pnt_left, pnt_right, pnt_center, lane_angle)
    :return: curseg, curlane, lanedev_rad, lanedev_dist, lanedev_rdist, lanewidth, pnt_center, rad_center,
            pnt_left, pnt_right
    """

    data_t = make_numpy_array(data_t, keep_1dim=True)

    # Get car-info
    pos_i = data_t[1:4]  # x, y, theta(rad)
    # carlen = data_t[6]
    carwidth = data_t[5]
    seg_t = int(data_t[7])
    lane_t = int(data_t[8])

    # Check on which track this car is
    if seg_t == -1 or lane_t == -1:
        curseg, curlane = get_index_seglane(pos_i, sim_track.pnts_poly_track)
    else:
        curseg, curlane = np.array([seg_t], dtype=np.int32), np.array([lane_t], dtype=np.int32)

    curseg_sel, curlane_sel = curseg[0], curlane[0]

    # Find feature
    # A. GET LANE DEVIATION DIST
    if curseg_sel == -1 or curlane_sel == -1:
        # Out of track
        # print("Out of track")
        lanedev_rad, lanedev_dist, lanedev_rdist, lanewidth = 0, 0, 0.5, 1
        pnt_center = np.array([0, 0], dtype=np.float32)
        rad_center = 0
        pnt_left = np.array([0, 0], dtype=np.float32)
        pnt_right = np.array([0, 0], dtype=np.float32)
    else:
        pnts_lr_border_lane = sim_track.pnts_lr_border_track[curseg_sel][curlane_sel]
        pnts_left = pnts_lr_border_lane[0]
        pnts_right = pnts_lr_border_lane[1]
        if use_intp == 0:
            pnt_left, _ = get_closest_pnt(pos_i[0:2], pnts_left)
            pnt_right, _ = get_closest_pnt(pos_i[0:2], pnts_right)
            vec_l2r = pnt_right[0:2] - pnt_left[0:2]
            rad = math.atan2(vec_l2r[1], vec_l2r[0])
            rad_center = (rad + math.pi / 2)
        else:
            pnt_left, _ = get_closest_pnt_intp(pos_i[0:2], pnts_left, num_intp=100)
            pnt_right, _ = get_closest_pnt_intp(pos_i[0:2], pnts_right, num_intp=100)
            vec_l2r = pnt_right[0:2] - pnt_left[0:2]

            type_tmp = sim_track.lane_type[curseg_sel][curlane_sel]
            if type_tmp == "Straight" or type_tmp == "straight":
                delta_sel = 5  # 5
            else:
                delta_sel = 1

            rad_center = get_lane_rad_wrt_mtrack(sim_track, pos_i[0:2], curseg_sel, curlane_sel, delta=delta_sel)

        lanedev_rad = pos_i[2] - rad_center
        lanedev_rad = angle_handle(lanedev_rad)
        pnt_center = (pnt_left + pnt_right) / 2

        vec_c2i = pos_i[0:2] - pnt_center[0:2]

        lanewidth = norm(vec_l2r[0:2])
        lanedev_dist = norm(vec_c2i[0:2]) * np.sign(vec_c2i[0]*vec_l2r[0] + vec_c2i[1]*vec_l2r[1])
        lanedev_rdist = norm(pos_i[0:2] - pnt_right[0:2])
        lanedev_rdist = lanedev_rdist - carwidth / 2

    return curseg, curlane, lanedev_rad, lanedev_dist, lanedev_rdist, lanewidth, pnt_center, rad_center, pnt_left, \
           pnt_right


def get_feature_sub2(sim_track, pos_i, seg_i, lane_i, data_ov, num_car, max_dist):
    """ Gets feature (sub2).
    :param sim_track: track-info
    :param pos_i: pose
    :param seg_i: segment index
    :param lane_i: lane index
    :param data_ov: other vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10)
    :param num_car: number of vehicle
    :param max_dist: maximum distance
    :return: ...
    """
    pnt_min = sim_track.pnt_min
    pnt_init = np.array([pnt_min[0] - 100, pnt_min[1] - 100], dtype=np.float32)

    th_lane_connected_lower = sim_track.th_lane_connected_lower
    th_lane_connected_upper = sim_track.th_lane_connected_upper

    d_lf, d_lr = max_dist, max_dist
    pnt_lf, pnt_lr = np.copy(pnt_init), np.copy(pnt_init)

    d_rf, d_rr = max_dist, max_dist
    pnt_rf, pnt_rr = np.copy(pnt_init), np.copy(pnt_init)

    d_cf, d_cr = max_dist, max_dist
    pnt_cf, pnt_cr = np.copy(pnt_init), np.copy(pnt_init)

    # Near vehicle id
    id_lf, id_lr, id_rf, id_rr, id_cf, id_cr = -1, -1, -1, -1, -1, -1

    # Near vehicle indexes
    idx_lf, idx_lr, idx_rf, idx_rr, idx_cf, idx_cr = -1, -1, -1, -1, -1, -1

    if seg_i == -1 or lane_i == -1 or num_car == 0:
        # Out of track
        # print("Out of track")
        pass
    else:
        track_dir_cur = sim_track.lane_dir[seg_i][lane_i]

        # B-0. FIND 'cpnt_c'
        pnts_lr_border_lane_c = sim_track.pnts_lr_border_track[seg_i][lane_i]
        pnts_left_c = pnts_lr_border_lane_c[0]
        pnts_right_c = pnts_lr_border_lane_c[1]

        lpnt_c, _ = get_closest_pnt(pos_i[0:2], pnts_left_c)
        rpnt_c, _ = get_closest_pnt(pos_i[0:2], pnts_right_c)
        cpnt_c = (lpnt_c + rpnt_c) / 2

        # B-1. LEFT FRONTAL AND REARWARD DISTANCE
        cond_not_leftmost = (lane_i > 0) if track_dir_cur == +1 else (lane_i < (sim_track.num_lane[seg_i] - 1))
        if cond_not_leftmost:  # NOT LEFTMOST LANE
            lane_i_left = (lane_i - 1) if track_dir_cur == +1 else (lane_i + 1)
            pnts_lr_border_lane_l = sim_track.pnts_lr_border_track[seg_i][lane_i_left]
            pnts_left_l = pnts_lr_border_lane_l[0]
            pnts_right_l = pnts_lr_border_lane_l[1]

            lpnt_l, _ = get_closest_pnt(pos_i[0:2], pnts_left_l)
            rpnt_l, _ = get_closest_pnt(pos_i[0:2], pnts_right_l)

            cpnt_l = (lpnt_l + rpnt_l) / 2
            vec_l2c = cpnt_c[0:2] - cpnt_l[0:2]
            norm_vec_l2c = norm(vec_l2c)

            if (norm_vec_l2c < th_lane_connected_upper) and (norm_vec_l2c > th_lane_connected_lower):
                # CHECK IF LEFT LANE IS CONNECTED
                vec_l2r = rpnt_l - lpnt_l
                rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
                forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)

                for j in range(0, num_car):
                    pos_j = data_ov[j, 1:4]  # x, y, theta
                    rx_j = data_ov[j, 6] / 2
                    lane_j = data_ov[j, 8]
                    id_j = data_ov[j, -1]

                    if abs(lane_j - lane_i_left) < 0.5:  # FOR THE LANE LEFT
                        vec_c2j = pos_j[0:2] - cpnt_l
                        dist_c2j = norm(vec_c2j)
                        dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                        if dot_tmp > 0:  # FRONTAL
                            if dist_c2j < d_lf:
                                alpha_tmp = (vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                pnt_lf = cpnt_l[0:2] + alpha_tmp * forwardvec[0:2]
                                d_lf = norm(cpnt_l[0:2] - pnt_lf[0:2]) * np.sign(alpha_tmp)
                                d_lf = min(d_lf, max_dist)
                                id_lf = id_j
                                idx_lf = j

                        else:  # REARWARD
                            if dist_c2j < d_lr:
                                alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                pnt_lr = cpnt_l[0:2] + alpha_tmp * (-forwardvec[0:2])
                                d_lr = norm(cpnt_l[0:2] - pnt_lr[0:2]) * np.sign(alpha_tmp)
                                d_lr = min(d_lr, max_dist)
                                id_lr = id_j
                                idx_lr = j
            else:
                d_lf, d_lr = 0, 0
        else:
            d_lf, d_lr = 0, 0

        # B-2. RIGHT FRONTAL AND REARWARD DISTANCE
        cond_not_rightmost = (lane_i < (sim_track.num_lane[seg_i] - 1)) if track_dir_cur == +1 else \
            (lane_i > 0)
        if cond_not_rightmost:  # NOT RIGHTMOST LANE
            lane_i_right = (lane_i + 1) if track_dir_cur == +1 else (lane_i - 1)
            pnts_lr_border_lane_r = sim_track.pnts_lr_border_track[seg_i][lane_i_right]
            pnts_left_r = pnts_lr_border_lane_r[0]
            pnts_right_r = pnts_lr_border_lane_r[1]

            lpnt_r, _ = get_closest_pnt(pos_i[0:2], pnts_left_r)
            rpnt_r, _ = get_closest_pnt(pos_i[0:2], pnts_right_r)
            cpnt_r = (lpnt_r + rpnt_r) / 2
            vec_r2c = cpnt_c[0:2] - cpnt_r[0:2]
            norm_vec_r2c = norm(vec_r2c)

            if (norm_vec_r2c < th_lane_connected_upper) and (norm_vec_r2c > th_lane_connected_lower):
                # CHECK IF RIGHT LANE IS CONNECTED
                vec_l2r = rpnt_r[0:2] - lpnt_r[0:2]
                rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
                forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
                for j in range(0, num_car):  # FOR ALL OTHER CARS
                    pos_j = data_ov[j, 1:4]  # x, y, theta
                    rx_j = data_ov[j, 6] / 2
                    lane_j = data_ov[j, 8]
                    id_j = data_ov[j, -1]

                    if abs(lane_j - lane_i_right) < 0.5:  # FOR THE LANE RIGHT
                        vec_c2j = pos_j[0:2] - cpnt_r[0:2]
                        dist_c2j = norm(vec_c2j)
                        dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                        if dot_tmp > 0:  # FRONTAL
                            if dist_c2j < d_rf:
                                alpha_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1] - rx_j
                                pnt_rf = cpnt_r[0:2] + alpha_tmp * forwardvec[0:2]
                                d_rf = norm(cpnt_r[0:2] - pnt_rf[0:2]) * np.sign(alpha_tmp)
                                d_rf = min(d_rf, max_dist)
                                id_rf = id_j
                                idx_rf = j

                        else:  # REARWARD
                            if dist_c2j < d_rr:
                                alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                                pnt_rr = cpnt_r[0:2] + alpha_tmp * (-forwardvec[0:2])
                                d_rr = norm(cpnt_r[0:2] - pnt_rr[0:2]) * np.sign(alpha_tmp)
                                d_rr = min(d_rr, max_dist)
                                id_rr = id_j
                                idx_rr = j
            else:
                d_rf, d_rr = 0, 0
        else:
            d_rf, d_rr = 0, 0

        # B-3. CENTER FRONTAL AND REARWARD DISTANCE
        vec_l2r = rpnt_c[0:2] - lpnt_c[0:2]
        rad = math.atan2(vec_l2r[1], vec_l2r[0]) + math.pi / 2
        forwardvec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)

        for j in range(0, num_car):  # FOR ALL OTHER CARS
            pos_j = data_ov[j, 1:4]  # x, y, theta
            rx_j = data_ov[j, 6] / 2
            lane_j = data_ov[j, 8]
            id_j = data_ov[j, -1]

            if abs(lane_j - lane_i) < 0.5:  # FOR THE SAME LANE
                vec_c2j = pos_j[0:2] - cpnt_c[0:2]
                dist_c2j = norm(vec_c2j)
                dot_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]
                if dot_tmp > 0:  # FRONTAL
                    if dist_c2j < d_cf:
                        alpha_tmp = vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1] - rx_j
                        pnt_cf = cpnt_c[0:2] + alpha_tmp * forwardvec[0:2]
                        d_cf = norm(cpnt_c[0:2] - pnt_cf[0:2]) * np.sign(alpha_tmp)
                        d_cf = min(d_cf, max_dist)
                        id_cf = id_j
                        idx_cf = j

                else:  # REARWARD
                    if dist_c2j < d_cr:
                        alpha_tmp = -(vec_c2j[0] * forwardvec[0] + vec_c2j[1] * forwardvec[1]) - rx_j
                        pnt_cr = cpnt_c[0:2] + alpha_tmp * (-forwardvec[0:2])
                        d_cr = norm(cpnt_c[0:2] - pnt_cr[0:2]) * np.sign(alpha_tmp)
                        d_cr = min(d_cr, max_dist)
                        id_cr = id_j
                        idx_cr = j

    return id_lf, id_lr, id_rf, id_rr, id_cf, id_cr, idx_lf, idx_lr, idx_cf, idx_cr, idx_rf, idx_rr, d_lf, d_lr, d_rf, \
           d_rr, d_cf, d_cr, pnt_lf, pnt_lr, pnt_rf, pnt_rr, pnt_cf, pnt_cr


def get_discretized_f(f_in, idx_f_in):
    """ Gets discretized feature (part).
    :param f_in: input feature (dim = N x 21)
    :param idx_f_in: input index of feature
    """

    f_in = make_numpy_array(f_in, keep_1dim=False)
    idx_f_in = make_numpy_array(idx_f_in, keep_1dim=True)
    num_data = f_in.shape[0]

    ld_dist_range = np.linspace(-0.5, +0.5, num=5)
    d_scaled_range = np.linspace(0, +30, num=10)
    ld_dist_range = np.reshape(ld_dist_range, (1, -1))
    d_scaled_range = np.reshape(d_scaled_range, (1, -1))

    f_new = np.copy(f_in)

    for nidx_f in range(0, idx_f_in.shape[0]):
        idx_f_sel = idx_f_in[nidx_f]
        f_new_sel = f_in[:, nidx_f]
        f_new_sel = np.reshape(f_new_sel, (-1, 1))

        if idx_f_sel == 4:
            val_range = np.copy(ld_dist_range)
        elif 6 <= idx_f_sel <= 11:
            val_range = np.copy(d_scaled_range)
        elif 12 <= idx_f_sel <= 17:
            val_range = np.copy(ld_dist_range)
        else:
            val_range = np.copy(ld_dist_range)
        val_range_t = np.tile(val_range, [num_data, 1])

        f_new_sel = np.tile(f_new_sel, [1, val_range_t.shape[1]])
        diff_tmp = np.abs(f_new_sel - val_range_t)
        idx_min = np.argmin(diff_tmp, axis=1)

        val_range_s = np.reshape(val_range, -1)
        val_tmp = val_range_s[idx_min]
        f_new[:, nidx_f] = val_tmp

    return f_new


def get_dist2goal(pnt_t, seg_t, lane_t, indexes_goal, pnts_goal):
    """ Gets distance to goal.
    :param pnt_t: target point (ndarray, dim = 2)
    :param seg_t: target seg index (int)
    :param lane_t: target lane index (int)
    :param indexes_goal: goal indexes - seg, lane (dim = N x 2)
    :param pnts_goal: goal points (dim = N x 2)
    """
    pnt_t = make_numpy_array(pnt_t, keep_1dim=True)

    max_dist = 10000

    if len(indexes_goal) > 0 and len(pnts_goal) > 0:
        num_goal_pnts = indexes_goal.shape[0]

        dist2goal_array = max_dist * np.ones((num_goal_pnts,), dtype=np.float32)
        reach_goal_array = np.zeros((num_goal_pnts,), dtype=np.int32)

        if int(seg_t) == -1 or int(lane_t) == -1:
            # print("Invalid segment & lane indexes")
            pass
        else:
            for nidx_d in range(0, num_goal_pnts):
                index_goal_sel = indexes_goal[nidx_d, :]
                pnt_goal_sel = pnts_goal[nidx_d, :]
                if seg_t == index_goal_sel[0] and lane_t == index_goal_sel[1]:
                    diff_sel = pnt_t[0:2] - pnt_goal_sel[0:2]
                    dist_sel = norm(diff_sel)

                    dist2goal_array[nidx_d] = min(dist_sel, max_dist)
                    if dist_sel < 6:
                        reach_goal_array[nidx_d] = 1
    else:
        dist2goal_array = max_dist * np.ones((1,), dtype=np.float32)
        reach_goal_array = np.zeros((1,), dtype=np.int32)

    return dist2goal_array, reach_goal_array


# ENCODE & DECODE TRAJECTORY ------------------------------------------------------------------------------------------#
def encode_traj(traj_in, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0):
    """ Encodes trajectory.
    :param traj_in: trajectory (dim = (H+1) x 3)
    :param traj_type: 0 --> prev: (1):t-H --> (end):t
                      1 --> post: (1):t --> (end):t+H
    :param pnts_poly_track: track-points (polygon)
    :param pnts_lr_border_track: track-points (left-right)
    :param is_track_simple: whether track is simple (highD) (boolean)
    """
    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    h = traj_in.shape[0] - 1

    traj_encoded = np.zeros((3 * h, ), dtype=np.float32)

    for nidx_h in range(0, h):
        if traj_type == 0:  # prev
            t_cur = h - nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur - 1, 0:3]
        else:  # post
            t_cur = nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur + 1, 0:3]

        if is_track_simple == 1:
            if abs(pnt_cur[2]) > math.pi / 2:
                rad_center_cur = math.pi
            else:
                rad_center_cur = 0.0
        else:
            _, rad_center_cur = get_lane_cp_angle(pnt_cur, pnts_poly_track, pnts_lr_border_track)

        diff_vec = pnt_next[0:2] - pnt_cur[0:2]
        dist_vec = norm(diff_vec)
        angle_diff = math.atan2(diff_vec[1], diff_vec[0]) - rad_center_cur
        dist_horiz = dist_vec * math.cos(angle_diff)
        dist_vertical = dist_vec * math.sin(angle_diff)
        diff_rad = pnt_next[2] - pnt_cur[2]

        idx_update = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        traj_encoded[idx_update] = [dist_horiz, dist_vertical, diff_rad]

    return traj_encoded


def encode_traj_2(traj_in, traj_type, sim_track, is_track_simple=0):
    """ Encodes trajectory.
    :param traj_in: trajectory (dim = (H+1) x 3)
    :param traj_type: 0 --> prev: (1):t-H --> (end):t
                      1 --> post: (1):t --> (end):t+H
    :param sim_track: track-info
    :param is_track_simple: whether track is simple (highD) (boolean)
    """
    traj_in = make_numpy_array(traj_in, keep_1dim=False)
    h = traj_in.shape[0] - 1

    traj_encoded = np.zeros((3 * h, ), dtype=np.float32)

    for nidx_h in range(0, h):
        if traj_type == 0:  # prev
            t_cur = h - nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur - 1, 0:3]
        else:  # post
            t_cur = nidx_h
            pnt_cur = traj_in[t_cur, 0:3]
            pnt_next = traj_in[t_cur + 1, 0:3]

        if is_track_simple == 1:
            if abs(pnt_cur[2]) > math.pi / 2:
                rad_center_cur = math.pi
            else:
                rad_center_cur = 0.0
        else:
            rad_center_cur = get_lane_rad_wrt_mtrack(sim_track, pnt_cur[0:2], -1, -1, delta=5)

        diff_vec = pnt_next[0:2] - pnt_cur[0:2]
        dist_vec = norm(diff_vec)
        angle_diff = math.atan2(diff_vec[1], diff_vec[0]) - rad_center_cur
        dist_horiz = dist_vec * math.cos(angle_diff)
        dist_vertical = dist_vec * math.sin(angle_diff)
        diff_rad = pnt_next[2] - pnt_cur[2]

        idx_update = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        traj_encoded[idx_update] = [dist_horiz, dist_vertical, diff_rad]

    return traj_encoded


def decode_traj(pos_in, val_in, h, traj_type, pnts_poly_track, pnts_lr_border_track, is_track_simple=0, dim_p=2):
    """ Decodes trajectory.
    :param pos_in: start pose (dim = 3)
    :param val_in: encoded val (dim = 3 * h)
    :param h: horizon-length (int)
    :param traj_type: 0 --> prev: (1):t-H --> (end):t
                      1 --> post: (1):t --> (end):t+H
    :param pnts_poly_track: track-points (polygon)
    :param pnts_lr_border_track: track-points (left-right)
    :param is_track_simple: whether track is simple (highD) (boolean)
    :param dim_p: dimension of point (2: heading is set to lane-angle)
    """
    pos_in = make_numpy_array(pos_in, keep_1dim=True)
    val_in = make_numpy_array(val_in, keep_1dim=True)

    traj_decoded = np.zeros((h + 1, 3), dtype=np.float32)

    pos_init = np.copy(pos_in)
    if is_track_simple == 1:
        if abs(pos_init[2]) > math.pi / 2:
            lanerad_init = math.pi
        else:
            lanerad_init = 0.0
    else:
        _, lanerad_init = get_lane_cp_angle(pos_init, pnts_poly_track, pnts_lr_border_track)
    devrad_init = pos_in[2] - lanerad_init

    traj_decoded[0, :] = pos_init
    pos_cur, devrad_cur, lanerad_cur = np.copy(pos_init), np.copy(devrad_init), np.copy(lanerad_init)
    pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_cur), np.copy(devrad_cur), np.copy(lanerad_cur)

    for nidx_h in range(0, h):
        pos_next = np.zeros((3, ), dtype=np.float32)

        idx_h = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        diff2next = val_in[idx_h]
        dist2next = norm(diff2next[0:2])

        # Find next position (x, y)
        angle_tmp1 = math.atan2(diff2next[1], diff2next[0])
        angle_tmp2 = lanerad_cur_tmp + angle_tmp1
        dx_tmp = dist2next * math.cos(angle_tmp2)
        dy_tmp = dist2next * math.sin(angle_tmp2)
        pos_next[0:2] = pos_cur_tmp[0:2] + np.array([dx_tmp, dy_tmp], dtype=np.float32)

        # Find next angle (theta)
        devrad_next = devrad_cur_tmp + diff2next[2]
        if is_track_simple == 1:
            lanerad_next = lanerad_init
        else:
            _, lanerad_next = get_lane_cp_angle(pos_next[0:2], pnts_poly_track, pnts_lr_border_track)

        if dim_p == 2:
            pos_next[2] = lanerad_next
        else:
            pos_next[2] = devrad_next + lanerad_next
        traj_decoded[nidx_h + 1, :] = pos_next

        # Update pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp
        pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_next), np.copy(devrad_next), np.copy(lanerad_next)

    if traj_type == 0:
        traj_decoded = np.flipud(traj_decoded)

    return traj_decoded


def decode_traj_2(pos_in, val_in, h, traj_type, sim_track, is_track_simple=0, dim_p=2):
    """ Decodes trajectory.
    :param pos_in: start pose (dim = 3)
    :param val_in: encoded val (dim = 3 * h)
    :param h: horizon-length (int)
    :param traj_type: 0 --> prev: (1):t-H --> (end):t
                      1 --> post: (1):t --> (end):t+H
    :param sim_track: track-info
    :param is_track_simple: whether track is simple (highD) (boolean)
    :param dim_p: dimension of point (2: heading is set to lane-angle)
    """
    pos_in = make_numpy_array(pos_in, keep_1dim=True)
    val_in = make_numpy_array(val_in, keep_1dim=True)

    traj_decoded = np.zeros((h + 1, 3), dtype=np.float32)

    pos_init = np.copy(pos_in)
    if is_track_simple == 1:
        if abs(pos_init[2]) > math.pi / 2:
            lanerad_init = math.pi
        else:
            lanerad_init = 0.0
    else:
        lanerad_init = get_lane_rad_wrt_mtrack(sim_track, pos_init[0:2], -1, -1, delta=5)
    devrad_init = pos_in[2] - lanerad_init

    traj_decoded[0, :] = pos_init
    pos_cur, devrad_cur, lanerad_cur = np.copy(pos_init), np.copy(devrad_init), np.copy(lanerad_init)
    pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_cur), np.copy(devrad_cur), np.copy(lanerad_cur)

    for nidx_h in range(0, h):
        pos_next = np.zeros((3, ), dtype=np.float32)

        idx_h = np.arange(3 * nidx_h, 3 * nidx_h + 3)
        diff2next = val_in[idx_h]
        dist2next = norm(diff2next[0:2])

        # Find next position (x, y)
        angle_tmp1 = math.atan2(diff2next[1], diff2next[0])
        angle_tmp2 = lanerad_cur_tmp + angle_tmp1
        dx_tmp = dist2next * math.cos(angle_tmp2)
        dy_tmp = dist2next * math.sin(angle_tmp2)
        pos_next[0:2] = pos_cur_tmp[0:2] + np.array([dx_tmp, dy_tmp], dtype=np.float32)

        # Find next angle (theta)
        devrad_next = devrad_cur_tmp + diff2next[2]
        if is_track_simple == 1:
            lanerad_next = lanerad_init
        else:
            lanerad_next = get_lane_rad_wrt_mtrack(sim_track, pos_next[0:2], -1, -1, delta=5)

        if dim_p == 2:
            pos_next[2] = lanerad_next
        else:
            pos_next[2] = devrad_next + lanerad_next
        traj_decoded[nidx_h + 1, :] = pos_next

        # Update pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp
        pos_cur_tmp, devrad_cur_tmp, lanerad_cur_tmp = np.copy(pos_next), np.copy(devrad_next), np.copy(lanerad_next)

    if traj_type == 0:
        traj_decoded = np.flipud(traj_decoded)

    return traj_decoded


# UTILS FOR TEST ------------------------------------------------------------------------------------------------------#
def set_initstate(data_ov, t, rx, ry, segidx, laneidx, margin_rx, margin_ry, track):
    """
    Sets intial states.
    :param data_ov: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
    :param t: time (float)
    :param rx: vehicle length (float)
    :param ry: vehicle width (float)
    :param segidx: segment index (int)
    :param laneidx: lane index (int)
    :param margin_rx: margin-x (float)
    :param margin_ry: margin-y (float)
    :param track: track-info
    """
    if segidx == -1:
        segidx = 0
    if laneidx == -1:
        laneidx = np.random.randint(3)

    if len(data_ov) > 0:  # EXIST OTHER-VEHICLES
        data_ov = make_numpy_array(data_ov, keep_1dim=False)

        # Select data (w.r.t time)
        idx_sel_ = np.where(data_ov[:, 0] == t)
        idx_sel = idx_sel_[0]
        data_vehicle_ov_sel = data_ov[idx_sel, :]

        # Select data (w.r.t lane)
        idx_sel_2 = []
        idx_sel_c_ = np.where(data_vehicle_ov_sel[:, 8] == laneidx)
        if len(idx_sel_c_) > 0:
            idx_sel_c = idx_sel_c_[0]
            idx_sel_2 = idx_sel_c

        if laneidx > 0:
            idx_sel_l_ = np.where(data_vehicle_ov_sel[:, 8] == (laneidx - 1))
            if len(idx_sel_l_) > 0:
                idx_sel_l = idx_sel_l_[0]
                idx_sel_2 = np.concatenate((idx_sel_2, idx_sel_l), axis=0)

        if laneidx < (track.num_lane[segidx] - 1):
            idx_sel_r_ = np.where(data_vehicle_ov_sel[:, 8] == (laneidx + 1))
            if len(idx_sel_r_) > 0:
                idx_sel_r = idx_sel_r_[0]
                idx_sel_2 = np.concatenate((idx_sel_2, idx_sel_r), axis=0)

        if len(idx_sel_2) > 0:
            idx_sel_2 = np.unique(idx_sel_2, axis=0)
            data_vehicle_ov_sel = data_vehicle_ov_sel[idx_sel_2, :]

        # Points (middle)
        pnts_c_intp = track.pnts_m_track[segidx][laneidx]

        # Modify points (middle)
        if track.trackname == "US101" or track.trackname == "I80" or track.trackname == "us101" or \
                track.trackname == "i80":
            if segidx == 0:
                len_pnts = pnts_c_intp.shape[0]
                idx_2_update = range(math.floor(len_pnts * 1 / 5), len_pnts)
                pnts_c_intp = pnts_c_intp[idx_2_update, :]
        elif "highD" in track.trackname or "highd" in track.trackname:
            len_pnts = pnts_c_intp.shape[0]
            idx_2_update = range(4, math.floor(len_pnts * 1 / 2))
            pnts_c_intp = pnts_c_intp[idx_2_update, :]
        else:
            len_pnts = pnts_c_intp.shape[0]
            idx_2_update = range(4, math.floor(len_pnts * 1 / 2))
            pnts_c_intp = pnts_c_intp[idx_2_update, :]

        # Check collision
        rad_c = 0
        idx_valid = np.zeros((pnts_c_intp.shape[0], ), dtype=np.int32)
        rad_valid = np.zeros((pnts_c_intp.shape[0], ), dtype=np.float64)
        cnt_valid = 0
        for nidx_p in range(0, pnts_c_intp.shape[0]):
            pnt_c_cur = pnts_c_intp[nidx_p, :]
            pnt_c_cur = pnt_c_cur.reshape(-1)

            if nidx_p < (pnts_c_intp.shape[0] - 1):
                pnt_c_next_ = pnts_c_intp[nidx_p + 1, :]
                pnt_c_next = pnt_c_next_.reshape(-1)
                vec_cur2next = pnt_c_next - pnt_c_cur
                rad_c = math.atan2(vec_cur2next[1], vec_cur2next[0])

            # data_t: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_t = [0, pnt_c_cur[0], pnt_c_cur[1], rad_c, 0, ry + margin_ry, rx + margin_rx, segidx, laneidx, -1]
            is_collision_out = check_collision(data_t, data_vehicle_ov_sel)

            if is_collision_out == 0:
                cnt_valid = cnt_valid + 1
                idx_valid[cnt_valid - 1], rad_valid[cnt_valid - 1] = nidx_p, rad_c

        idx_valid, rad_valid = idx_valid[0:cnt_valid], rad_valid[0:cnt_valid]

        if cnt_valid > 0:
            if segidx == 0:
                idx_rand_0 = round(cnt_valid / 4)
                idx_rand = idx_rand_0 + np.random.randint(cnt_valid - idx_rand_0)
            else:
                idx_rand = np.random.randint(cnt_valid)
            idx_valid_sel = idx_valid[idx_rand]

            pnt_sel, rad_sel = pnts_c_intp[idx_valid_sel, :], rad_valid[idx_rand]

            # data_sel: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_sel = [0, pnt_sel[0], pnt_sel[1], rad_sel, 0, ry, rx, segidx, laneidx, -1]
            _, _, _, _, _, _, pnt_center, rad_center, _, _ = get_feature_sub1(track, data_sel, use_intp=0)

            pnts_init = pnts_c_intp[idx_valid, :]

            succeed_init = 1
        else:
            succeed_init = 0
    else:  # NO OTHER-VEHICLES
        succeed_init = 1

        # Points (middle)
        pnts_c_intp = track.pnts_m_track[segidx][laneidx]

        idx_rand = np.random.randint(pnts_c_intp.shape[0] - 1)
        pnt_sel = pnts_c_intp[idx_rand, :]
        pnt_next = pnts_c_intp[idx_rand + 1, :]
        vec_cur2next = pnt_next - pnt_sel
        rad_sel = math.atan2(vec_cur2next[1], vec_cur2next[0])

        # data_sel: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
        data_sel = [0, pnt_sel[0], pnt_sel[1], rad_sel, 0, ry, rx, segidx, laneidx, -1]
        _, _, _, _, _, _, pnt_center, rad_center, _, _ = get_feature_sub1(track, data_sel, use_intp=0)

        pnts_init = pnts_c_intp[np.arange(0, pnts_c_intp.shape[0] - 1), :]

    # Update
    if succeed_init == 1:
        x_init, y_init, theta_init = pnt_center[0], pnt_center[1], rad_center
    else:
        x_init, y_init, theta_init = [], [], []
        pnts_init = []

    return succeed_init, x_init, y_init, theta_init, pnts_init


def set_initstate_control(id_ev, v_ref, ry_ev, rx_ev, seg_init_list, lane_init_list, data_v, t_min, t_max,
                          mode_init, track, is_track_simple=0):
    """
    Sets initial-state of vehicle for control.
    """
    if mode_init == 0:  # Add new vehicle
        t_init = t_min + 20 + np.random.randint(int((t_max - t_min - 20) * 3 / 5))  # Starting time
        t_init = int(t_init)
        id_ev = -1  # Id of ego-vehicle
        t_horizon = min(500, t_max - t_init - 1)  # Time horizon

        # Insert ego-vehicle
        idx_seg_rand, idx_lane_rand = np.random.randint(len(seg_init_list)), np.random.randint(len(lane_init_list))
        seg_init, lane_init = seg_init_list[idx_seg_rand], lane_init_list[idx_lane_rand]  # Initial indexes
        margin_rx_init, margin_ry_init = rx_ev, 0.2
        succeed_init, x_init, y_init, theta_init, pnts_init = \
            set_initstate(data_v, t_init, rx_ev, ry_ev, seg_init, lane_init, margin_rx_init, margin_ry_init, track)

        if succeed_init == 0:
            print("[ERROR] FAILED TO INITSTATE!")
            data_ev_init, data_ev = [], []
        else:
            # Set data
            data_ev_init = np.array([t_init, x_init, y_init, theta_init, v_ref, ry_ev, rx_ev, seg_init, lane_init, id_ev],
                                    dtype=np.float32)
            data_ev_init_ = np.reshape(data_ev_init, (1, -1))
            data_ev = np.repeat(data_ev_init_, (t_horizon + 2), axis=0)
            data_ev[:, 0] = np.arange(t_init, t_init + t_horizon + 2)

        data_ov = select_data_trange(data_v, t_init - 20, t_init + t_horizon + 2)
    else:  # Remove existing vehicle and replace with new one
        if id_ev == -1:  # Select existing vehicle at random
            t_init = t_min + 20 + np.random.randint(int((t_max - t_min - 20) * 3 / 5))  # Starting time
            t_init = int(t_init)
            _, idx_init_t_sl = select_data_t_seglane(data_v, t_init, seg_init_list, lane_init_list)
            idx_init_sel = idx_init_t_sl[np.random.randint(len(idx_init_t_sl))]
            data_ev_init = np.reshape(data_v[idx_init_sel, :], -1)
            id_ev = data_ev_init[-1]
            idx_ev_ = np.where(data_v[:, -1] == id_ev)
            idx_ev = idx_ev_[0]
        else:  # Select vehicle with id 'id_ev'
            idx_ev_ = np.where(data_v[:, -1] == id_ev)
            idx_ev = idx_ev_[0]
            if len(idx_ev) == 0:
                print("[ERROR] VEHICLE-ID cannot be found.")

            data_v_in_ev = data_v[idx_ev, :]
            t_init_add_1 = 0
            for t_init_add_1 in range(0, 50):
                seg_tmp, lane_tmp = get_index_seglane(data_v_in_ev[0 + t_init_add_1, 1:3], track.pnts_poly_track)
                if seg_tmp != -1 and lane_tmp != -1:
                    break

            t_init_add_2 = 10 if is_track_simple == 0 else 1

            t_init = int(min(data_v_in_ev[:, 0])) + t_init_add_1 + t_init_add_2  # Starting time
            data_ev_init, _ = select_data_t_id(data_v, t_init, id_ev)
            data_ev_init = np.reshape(data_ev_init, -1)

        if ry_ev > 0 and rx_ev > 0:
            data_v[idx_ev, 4], data_v[idx_ev, 5], data_v[idx_ev, 6] = v_ref, ry_ev, rx_ev

        # Time horizon
        # t_horizon = min(700, int(np.amax(data_v[idx_ev, 0])) - t_init - 50)
        t_horizon = 1200

        # Set data
        _, data_ev, data_ov = select_data_id_trange(data_v, id_ev, t_init - 20, t_init + t_horizon + 1)

        len_tmp = (t_init + t_horizon + 1) - (t_init - 20) + 1
        if data_ev.shape[0] < len_tmp:
            len_tmp_add = len_tmp - data_ev.shape[0]
            data_ev_end = data_ev[-1, :]
            data_ev_end = np.reshape(data_ev_end, (1, -1))
            data_ev_add = np.tile(data_ev_end, (len_tmp_add, 1))
            data_ev_add[:, 0] = np.arange(data_ev_end[0,0]+1, data_ev_end[0,0]+len_tmp_add + 1)
            data_ev = np.concatenate((data_ev, data_ev_add), axis=0)

    return id_ev, t_init, t_horizon, data_ev_init, data_ev, data_ov


def set_initstate_pred(id_tv, ry_tv, rx_tv, data_v, track, is_track_simple=0):
    """
    Sets initial-state of vehicle for prediction.
    """
    # Remove existing vehicle and replace with new one
    # Select vehicle with id 'id_ev'
    idx_ev_ = np.where(data_v[:, -1] == id_tv)
    idx_ev = idx_ev_[0]
    if len(idx_ev) == 0:
        print("[ERROR] VEHICLE-ID cannot be found.")

    data_v_in_ev = data_v[idx_ev, :]
    t_init_add_1 = 0
    for t_init_add_1 in range(0, 50):
        seg_tmp, lane_tmp = get_index_seglane(data_v_in_ev[0 + t_init_add_1, 1:3], track.pnts_poly_track)
        if seg_tmp != -1 and lane_tmp != -1:
            break

    t_init_add_2 = 10 if is_track_simple == 0 else 1

    t_init = int(min(data_v_in_ev[:, 0])) + t_init_add_1 + t_init_add_2  # Starting time
    data_ev_init, _ = select_data_t_id(data_v, t_init, id_tv)
    data_ev_init = np.reshape(data_ev_init, -1)

    if ry_tv > 0 and rx_tv > 0:
        data_v[idx_ev, 5], data_v[idx_ev, 6] = ry_tv, rx_tv

    # Time horizon
    t_horizon = min(300, int(np.amax(data_v[idx_ev, 0])) - t_init - 50)

    # Set data
    _, data_ev, data_ov = select_data_id_trange(data_v, id_tv, t_init - 20, t_init + t_horizon + 51)

    return t_init, t_horizon, data_ev_init, data_ev, data_ov


def get_info_t(t_cur, id_ev, data_ev, data_ev_cur, data_ov_cur, idx_f_use, horizon_prev, horizon_post,
               idx_prev_sp, idx_post_sp, dim_p, sim_track, use_intp=0):
    """
    Gets current info.
    data-vehicle: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    """

    pnts_poly_track = sim_track.pnts_poly_track
    pnts_lr_border_track = sim_track.pnts_lr_border_track

    f_raw_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, rad_center, dist_cf = \
        get_feature(sim_track, data_ev_cur, data_ov_cur, use_intp=use_intp)
    ldev_rad_cur = f_raw_cur[0]  # Current lanedev-rad
    lanewidth_cur = f_raw_cur[3]  # Current lanewidth
    ldev_dist_scaled_cur = f_raw_cur[4]  # Current lanedev-dist (scaled)
    f_cur = f_raw_cur[idx_f_use] if len(idx_f_use) > 0 else np.copy(f_raw_cur)
    pnt_center, pnt_left, pnt_right = pnts_debug_f_ev[0, :], pnts_debug_f_ev[1, :], pnts_debug_f_ev[2, :]

    id_rest_cur = np.setdiff1d(np.unique(data_ov_cur[:, -1]), id_near_cur)  # ids of ov beside near

    prevtraj_ev, prevtraj_ev_enc, prevtraj_ev_sp, prevtraj_ev_enc_sp = [], [], [], []
    posttraj_ev, vposttraj_ev, posttraj_ev_enc, posttraj_ev_sp, posttraj_ev_enc_sp = [], [], [], [], []
    if horizon_prev > 0:
        prevtraj_ev, _ = get_vehicle_traj_per_id(data_ev, t_cur, id_ev, horizon_prev, do_reverse=1, handle_remain=1)
        _prevtraj_ev_enc = encode_traj(prevtraj_ev, 0, pnts_poly_track, pnts_lr_border_track)

        if dim_p == 2:
            _idx_tmp_prev_1 = np.arange(0, _prevtraj_ev_enc.shape[0])
            _idx_tmp_prev_2 = np.arange(dim_p, _prevtraj_ev_enc.shape[0], 3)
            _idx_tmp_prev_3 = np.setdiff1d(_idx_tmp_prev_1, _idx_tmp_prev_2)
            prevtraj_ev_enc = _prevtraj_ev_enc[_idx_tmp_prev_3]
        else:
            prevtraj_ev_enc = _prevtraj_ev_enc

        if len(idx_prev_sp) > 0:
            prevtraj_ev_sp = prevtraj_ev[idx_prev_sp, :]
            _prevtraj_ev_enc_sp = encode_traj(prevtraj_ev_sp, 0, pnts_poly_track, pnts_lr_border_track)
            if dim_p == 2:
                _idx_tmp_prev_sp_1 = np.arange(0, _prevtraj_ev_enc_sp.shape[0])
                _idx_tmp_prev_sp_2 = np.arange(dim_p, _prevtraj_ev_enc_sp.shape[0], 3)
                _idx_tmp_prev_sp_3 = np.setdiff1d(_idx_tmp_prev_sp_1, _idx_tmp_prev_sp_2)
                prevtraj_ev_enc_sp = _prevtraj_ev_enc_sp[_idx_tmp_prev_sp_3]
            else:
                prevtraj_ev_enc_sp = _prevtraj_ev_enc_sp

    if horizon_post > 0:
        posttraj_ev, _ = get_vehicle_traj_per_id(data_ev, t_cur, id_ev, horizon_post, do_reverse=0, handle_remain=1)
        vposttraj_ev = get_vehicle_vtraj_per_id(data_ev, t_cur, id_ev, horizon_post, do_reverse=0, handle_remain=1)
        _posttraj_ev_enc = encode_traj(posttraj_ev, 1, pnts_poly_track, pnts_lr_border_track)

        if dim_p == 2:
            _idx_tmp_post_1 = np.arange(0, _posttraj_ev_enc.shape[0])
            _idx_tmp_post_2 = np.arange(dim_p, _posttraj_ev_enc.shape[0], 3)
            _idx_tmp_post_3 = np.setdiff1d(_idx_tmp_post_1, _idx_tmp_post_2)
            posttraj_ev_enc = _posttraj_ev_enc[_idx_tmp_post_3]
        else:
            posttraj_ev_enc = _posttraj_ev_enc

        if len(idx_post_sp) > 0:
            posttraj_ev_sp = posttraj_ev[idx_post_sp, :]
            _posttraj_ev_enc_sp = encode_traj(posttraj_ev_sp, 1, pnts_poly_track, pnts_lr_border_track)

            if dim_p == 2:
                _idx_tmp_post_sp_1 = np.arange(0, _posttraj_ev_enc_sp.shape[0])
                _idx_tmp_post_sp_2 = np.arange(dim_p, _posttraj_ev_enc_sp.shape[0], 3)
                _idx_tmp_post_sp_3 = np.setdiff1d(_idx_tmp_post_sp_1, _idx_tmp_post_sp_2)
                posttraj_ev_enc_sp = _posttraj_ev_enc_sp[_idx_tmp_post_sp_3]
            else:
                posttraj_ev_enc_sp = _posttraj_ev_enc_sp

    dict_out = {'f_raw': f_raw_cur, 'f': f_cur, 'dist_cf': dist_cf, 'id_near': id_near_cur, 'id_rest': id_rest_cur,
                'lanedev_rad': ldev_rad_cur, 'lanewidth': lanewidth_cur, 'lanedev_dist_scaled': ldev_dist_scaled_cur,
                'pnt_center': pnt_center, 'rad_center': rad_center, 'pnt_left': pnt_left, 'pnt_right': pnt_right,
                'x_ev': prevtraj_ev, 'x_ev_enc': prevtraj_ev_enc,
                'x_ev_sp': prevtraj_ev_sp, 'x_ev_enc_sp': prevtraj_ev_enc_sp,
                'y_ev': posttraj_ev, 'vy_ev': vposttraj_ev, 'y_ev_enc': posttraj_ev_enc,
                'y_ev_sp': posttraj_ev_sp, 'y_ev_enc_sp': posttraj_ev_enc_sp}

    return dict_out


def get_multi_info_t(t_cur, id_ev, data_ev, data_ov, data_ev_cur, data_ov_cur, idx_f_use, horizon_prev, horizon_post,
                     idx_prev_sp, idx_post_sp, dim_p, sim_track, use_intp=0):
    """
    Gets current info (include near other vehicles).
    data-vehicle: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    """
    pnts_poly_track = sim_track.pnts_poly_track
    pnts_lr_border_track = sim_track.pnts_lr_border_track

    dim_pretraj_enc = dim_p * horizon_prev
    dim_pretraj_enc_sp = dim_p * (len(idx_prev_sp) - 1)
    dim_posttraj_enc = dim_p * horizon_post
    dim_posttraj_enc_sp = dim_p * (len(idx_post_sp) - 1)

    dict_info_t = get_info_t(t_cur, id_ev, data_ev, data_ev_cur, data_ov_cur, idx_f_use, horizon_prev, horizon_post,
                             idx_prev_sp, idx_post_sp, dim_p, sim_track, use_intp=use_intp)

    id_near = dict_info_t['id_near']

    prevtraj_ov = np.zeros((len(id_near), horizon_prev + 1, 3), dtype=np.float32)
    prevtraj_ov_enc = np.zeros((len(id_near), dim_pretraj_enc), dtype=np.float32)
    prevtraj_ov_sp = np.zeros((len(id_near), len(idx_prev_sp), 3), dtype=np.float32)
    prevtraj_ov_enc_sp = np.zeros((len(id_near), dim_pretraj_enc_sp), dtype=np.float32)

    posttraj_ov = np.zeros((len(id_near), horizon_post + 1, 3), dtype=np.float32)
    vposttraj_ov = np.zeros((len(id_near), horizon_post + 1), dtype=np.float32)
    posttraj_ov_enc = np.zeros((len(id_near), dim_posttraj_enc), dtype=np.float32)
    posttraj_ov_sp = np.zeros((len(id_near), len(idx_post_sp), 3), dtype=np.float32)
    posttraj_ov_enc_sp = np.zeros((len(id_near), dim_posttraj_enc_sp), dtype=np.float32)

    s_ov = np.zeros((len(id_near), 3), dtype=np.float32)
    size_ov = np.zeros((len(id_near), 2), dtype=np.float32)

    for nidx_near in range(0, len(id_near)):
        id_near_sel = id_near[nidx_near]

        prevtraj_ov_tmp = -1000 * np.ones((horizon_prev + 1, 3), dtype=np.float32)
        posttraj_ov_tmp = -1000 * np.ones((horizon_post + 1, 3), dtype=np.float32)
        vposttraj_ov_tmp = np.zeros((horizon_post + 1,), dtype=np.float32)

        prevtraj_ov_sp_tmp = prevtraj_ov_tmp[idx_prev_sp, :]
        posttraj_ov_sp_tmp = posttraj_ov_tmp[idx_post_sp, :]

        prevtraj_ov_enc_tmp = np.zeros((1, dim_pretraj_enc), dtype=np.float32)
        prevtraj_ov_enc_sp_tmp = np.zeros((1, dim_pretraj_enc_sp), dtype=np.float32)
        posttraj_ov_enc_tmp = np.zeros((1, dim_posttraj_enc), dtype=np.float32)
        posttraj_ov_enc_sp_tmp = np.zeros((1, dim_posttraj_enc_sp), dtype=np.float32)

        s_ov_tmp = -1000 * np.zeros((3,), dtype=np.float32)
        size_ov_tmp = np.zeros((2,), dtype=np.float32)
        if id_near_sel != -1:
            if horizon_prev > 0:
                prevtraj_ov_tmp, _ = get_vehicle_traj_per_id(data_ov, t_cur, id_near_sel, horizon_prev, do_reverse=1, handle_remain=1)
                _prevtraj_ov_enc_tmp = encode_traj(prevtraj_ov_tmp, 0, pnts_poly_track, pnts_lr_border_track)

                if dim_p == 2:
                    _idx_tmp_prev_1 = np.arange(0, _prevtraj_ov_enc_tmp.shape[0])
                    _idx_tmp_prev_2 = np.arange(dim_p, _prevtraj_ov_enc_tmp.shape[0], 3)
                    _idx_tmp_prev_3 = np.setdiff1d(_idx_tmp_prev_1, _idx_tmp_prev_2)
                    prevtraj_ov_enc_tmp = _prevtraj_ov_enc_tmp[_idx_tmp_prev_3]
                else:
                    prevtraj_ov_enc_tmp = _prevtraj_ov_enc_tmp

                if len(idx_prev_sp) > 0:
                    prevtraj_ov_sp_tmp = prevtraj_ov_tmp[idx_prev_sp, :]
                    _prevtraj_ov_enc_sp = encode_traj(prevtraj_ov_sp_tmp, 0, pnts_poly_track, pnts_lr_border_track)
                    if dim_p == 2:
                        _idx_tmp_prev_sp_1 = np.arange(0, _prevtraj_ov_enc_sp.shape[0])
                        _idx_tmp_prev_sp_2 = np.arange(dim_p, _prevtraj_ov_enc_sp.shape[0], 3)
                        _idx_tmp_prev_sp_3 = np.setdiff1d(_idx_tmp_prev_sp_1, _idx_tmp_prev_sp_2)
                        prevtraj_ov_enc_sp_tmp = _prevtraj_ov_enc_sp[_idx_tmp_prev_sp_3]
                    else:
                        prevtraj_ov_enc_sp_tmp = _prevtraj_ov_enc_sp

            if horizon_post > 0:
                posttraj_ov_tmp, size_ov_tmp = get_vehicle_traj_per_id(data_ov, t_cur, id_near_sel, horizon_post, do_reverse=0, handle_remain=1)
                vposttraj_ov_tmp = get_vehicle_vtraj_per_id(data_ov, t_cur, id_near_sel, horizon_post, do_reverse=0, handle_remain=1)
                _posttraj_ov_enc_tmp = encode_traj(posttraj_ov_tmp, 1, pnts_poly_track, pnts_lr_border_track)

                s_ov_tmp = posttraj_ov_tmp[0, 0:3]

                if dim_p == 2:
                    _idx_tmp_post_1 = np.arange(0, _posttraj_ov_enc_tmp.shape[0])
                    _idx_tmp_post_2 = np.arange(dim_p, _posttraj_ov_enc_tmp.shape[0], 3)
                    _idx_tmp_post_3 = np.setdiff1d(_idx_tmp_post_1, _idx_tmp_post_2)
                    posttraj_ov_enc_tmp = _posttraj_ov_enc_tmp[_idx_tmp_post_3]
                else:
                    posttraj_ov_enc_tmp = _posttraj_ov_enc_tmp

                if len(idx_post_sp) > 0:
                    posttraj_ov_sp_tmp = posttraj_ov_tmp[idx_post_sp, :]
                    _posttraj_ov_enc_sp_tmp = encode_traj(posttraj_ov_sp_tmp, 1, pnts_poly_track, pnts_lr_border_track)

                    if dim_p == 2:
                        _idx_tmp_post_sp_1 = np.arange(0, _posttraj_ov_enc_sp_tmp.shape[0])
                        _idx_tmp_post_sp_2 = np.arange(dim_p, _posttraj_ov_enc_sp_tmp.shape[0], 3)
                        _idx_tmp_post_sp_3 = np.setdiff1d(_idx_tmp_post_sp_1, _idx_tmp_post_sp_2)
                        posttraj_ov_enc_sp_tmp = _posttraj_ov_enc_sp_tmp[_idx_tmp_post_sp_3]
                    else:
                        posttraj_ov_enc_sp_tmp = _posttraj_ov_enc_sp_tmp

        prevtraj_ov[nidx_near, :, :] = prevtraj_ov_tmp
        prevtraj_ov_enc[nidx_near, :] = prevtraj_ov_enc_tmp
        prevtraj_ov_sp[nidx_near, :, :] = prevtraj_ov_sp_tmp
        prevtraj_ov_enc_sp[nidx_near, :] = prevtraj_ov_enc_sp_tmp

        posttraj_ov[nidx_near, :, :] = posttraj_ov_tmp
        vposttraj_ov[nidx_near, :] = vposttraj_ov_tmp
        posttraj_ov_enc[nidx_near, :] = posttraj_ov_enc_tmp
        posttraj_ov_sp[nidx_near, :, :] = posttraj_ov_sp_tmp
        posttraj_ov_enc_sp[nidx_near, :] = posttraj_ov_enc_sp_tmp

        s_ov[nidx_near, :] = s_ov_tmp
        size_ov[nidx_near, :] = size_ov_tmp

    dict_info_t_added = {'x_ov': prevtraj_ov, 'x_ov_enc': prevtraj_ov_enc, 'x_ov_sp': prevtraj_ov_sp,
                         'x_ov_enc_sp': prevtraj_ov_enc_sp, 'y_ov': posttraj_ov, 'vy_ov': vposttraj_ov,
                         'y_ov_enc': posttraj_ov_enc, 'y_ov_sp': posttraj_ov_sp, 'y_ov_enc_sp': posttraj_ov_enc_sp,
                         's_ov': s_ov, 'size_ov': size_ov}

    dict_info_t.update(dict_info_t_added)

    return dict_info_t


def recover_traj_pred(pose_ev, predval_y, y_mean, y_std, dim_p, h_y, sim_track, is_track_simple=0, xth=-1, yth=-1):
    """
    Recovers trajectory prediction.
    :param pose_ev: pose of ego-vehicle (x, y, theta) (dim = 3)
    :param predval_y: prediction on y [batch_size=1, timesteps, dim_p] or [timesteps, dim_p]
    :param y_mean: mean of y (dim = timesteps x dim_p)
    :param y_std: std of y (dim = timesteps x dim_p)
    :param h_y: time-steps (int)
    :param sim_track: track-info
    :param is_track_simple: whether track is simple (boolean)
    :xth: threshold on horizontal distance (-1: neglect) (float)
    :yth: threshold on vertical distance (-1: neglect) (float)
    """
    if len(y_mean) == 0 or len(y_std) == 0:
        do_recover = False
    else:
        do_recover = True

    predval_y = predval_y.reshape(1, -1)

    if do_recover:
        val_y_pred_rc = recover_from_normalized_data(predval_y, y_mean, y_std)
    else:
        val_y_pred_rc = predval_y

    # Vehicle cannot go back in a horizontal direction to the lane
    size_tmp = val_y_pred_rc.shape[1]
    idx_1_tmp = np.arange(0, size_tmp, dim_p)
    idx_2_tmp = np.where(val_y_pred_rc[0, idx_1_tmp] < 0.0)
    idx_2_tmp = idx_2_tmp[0]
    if len(idx_2_tmp) > 0:
        idx_3_tmp = idx_1_tmp[idx_2_tmp]
        val_y_pred_rc[0, idx_3_tmp] = 0

    # Vehicle cannot shift horizontally up to a certain level
    if xth > 0:
        idx_1_tmp = np.arange(0, size_tmp, dim_p)
        idx_2_tmp = np.where(np.abs(val_y_pred_rc[0, idx_1_tmp]) > xth)
        idx_2_tmp = idx_2_tmp[0]
        if len(idx_2_tmp) > 0:
            idx_3_tmp = idx_1_tmp[idx_2_tmp]
            val_y_pred_rc[0, idx_3_tmp] = xth

    # Vehicle cannot shift vertically up to a certain level
    if yth > 0:
        idx_1_tmp = np.arange(1, size_tmp, dim_p)
        idx_2_tmp = np.where(np.abs(val_y_pred_rc[0, idx_1_tmp]) > yth)
        idx_2_tmp = idx_2_tmp[0]
        if len(idx_2_tmp) > 0:
            idx_3_tmp = idx_1_tmp[idx_2_tmp]
            val_y_pred_rc[0, idx_3_tmp] = np.sign(val_y_pred_rc[0, idx_3_tmp]) * yth

    if dim_p == 2:
        val_y_pred_rc = val_y_pred_rc.reshape((h_y, dim_p))
        _zeros_tmp = np.zeros((h_y, 1), dtype=np.float32)
        val_y_pred_rc = np.concatenate((val_y_pred_rc, _zeros_tmp), axis=1)
        val_y_pred_rc = val_y_pred_rc.reshape(1, -1)

    traj_decoded = decode_traj_2(pose_ev[0:3], val_y_pred_rc, h_y, 1, sim_track, is_track_simple=is_track_simple,
                                 dim_p=dim_p)

    return traj_decoded


def check_reachgoal(pnt_ev, goal_pnts, d_th=1.0):
    """
    Checks the ego-vehicle reaches goal points.
    :param pnt_ev: ego-vehicle position (dim = 2)
    :param goal_pnts: goal positions (dim = N x 2)
    :param d_th: goal distance threshold
    """
    pnt_ev = make_numpy_array(pnt_ev, keep_1dim=True)
    pnt_ev = pnt_ev[0:2]
    pnt_ev_r = np.reshape(pnt_ev, (1, -1))

    goal_pnts = make_numpy_array(goal_pnts, keep_1dim=False)
    len_g = goal_pnts.shape[0]
    pnt_ev_rt = np.tile(pnt_ev_r, (len_g, 1))

    diff_tmp = pnt_ev_rt - goal_pnts
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    dist_g = np.reshape(dist_tmp, -1)
    dist_g_min = min(dist_g)

    is_reachgoal = True if dist_g_min < d_th else False

    return is_reachgoal


def set_data_test(len_data, data_ref):
    """
    Sets test vehicle data from reference data.
    """
    len_ref = data_ref.shape[0]

    if len_data > len_ref:
        data_out = np.zeros((len_data, 10), dtype=np.float32)
        len_dummy = data_out.shape[0] - len_ref
        data_out[0:data_ref.shape[0], :] = data_ref
        data_ev_end_ = np.reshape(data_ref[-1, :], (1, -1))
        data_out[data_ref.shape[0]:, :] = np.tile(data_ev_end_, (len_dummy, 1))
        data_out[data_ref.shape[0]:, 0] = np.arange(data_ev_end_[0, 0] + 1, data_ev_end_[0, 0] + len_dummy + 1)
    else:
        data_out = np.copy(data_ref)

    return data_out

def get_costmap_from_trajs(x_range, y_range, trajs, n_pnt=300, alpha_min=1.0, alpha_max=2.5):
    """
    Gets costmap form trajectories.
    """
    y_ref = np.zeros((20000, 2), dtype=np.float32)
    t_ref = np.zeros((20000,), dtype=np.float32)
    cnt_ref = 0
    for nidx_n in range(0, len(trajs)):
        _traj_sel = trajs[nidx_n]
        traj_sel = interpolate_traj(_traj_sel, alpha=2)
        idx_tmp = np.arange(cnt_ref, cnt_ref + traj_sel.shape[0])
        y_ref[idx_tmp, :] = traj_sel[:, 0:2]
        t_ref[idx_tmp] = np.arange(0, traj_sel.shape[0])
        cnt_ref = cnt_ref + traj_sel.shape[0]
    y_ref = y_ref[0:cnt_ref, :]
    t_ref = t_ref[0:cnt_ref]

    map_x0 = np.linspace(x_range[0], x_range[1], n_pnt)
    map_x1 = np.linspace(y_range[0], y_range[1], n_pnt)
    map_z = np.zeros((n_pnt, n_pnt), dtype=np.float32)

    def kernel_gaussian_mod(_x_0, _x_1, _xi, _ti, _alpha_min=2.0, _alpha_max=3.0):
        """ Modified version of Gaussian kernel. """
        _len_data = _xi.shape[0]
        _x_r = np.array([[_x_0, _x_1]], dtype=np.float32)
        _x_ext = np.tile(_x_r, [_len_data, 1])
        _diff = _x_ext - _xi

        _ti_min, _ti_max = min(_ti), max(_ti)
        _alpha = _alpha_min + (_alpha_max - _alpha_min) / (_ti_max - _ti_min) * _ti
        _alpha_r = np.reshape(_alpha,(-1, 1))
        _alpha_ext = np.repeat(_alpha_r, 2, axis=1)

        _diff_sq = np.square(_diff / _alpha_ext)
        _diff_sq_sum = np.sum(_diff_sq, axis=1)
        _exponent = -0.5 * _diff_sq_sum
        _k = np.exp(_exponent) * 1 / np.sqrt(2 * math.pi) / _alpha
        _k_out = np.sum(_k) / _len_data
        return _k_out

    for nidx_x0 in range(0, n_pnt):
        x0_sel = map_x0[nidx_x0]
        for nidx_x1 in range(0, n_pnt):
            x1_sel = map_x1[nidx_x1]
            k_out = kernel_gaussian_mod(x0_sel, x1_sel, y_ref, t_ref, _alpha_min=alpha_min, _alpha_max=alpha_max)
            map_z[nidx_x1, nidx_x0] = k_out

    return map_x0, map_x1, map_z
