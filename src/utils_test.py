# UTILITY FUNCTIONS FOR TEST

from __future__ import print_function

import numpy as np

from src.utils import *
from src.utils_sim import *

__all__ = ["get_current_info_multi", "get_current_info_part_near", "recover_traj_pred"]


def get_current_info_multi(t_cur, id_ev, data_ev, data_ov, data_ev_cur, data_ov_cur, idx_f_use, num_near, horizon_post,
                           horizon_prev, idx_post_sp, idx_prev_sp, dim_pretraj_sp, sim_track, use_intp=0):
    """
    Gets current info (include near other vehicles).
    data-vehicle: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    """

    f, id_near, pnts_debug_f_ev, pnts_debug_f_ov, rad_center, dist_cf = get_feature(sim_track, data_ev_cur,
                                                                                    data_ov_cur, use_intp=use_intp)
    lanewidth_cur = f[3]  # Current lanewidth
    f = f[idx_f_use]
    pnt_center, pnt_left, pnt_right = pnts_debug_f_ev[0, :], pnts_debug_f_ev[1, :], pnts_debug_f_ev[2, :]

    posttraj_ev, _ = get_vehicle_traj_per_id(data_ev, t_cur, id_ev, horizon_post, do_reverse=0, handle_remain=1)
    vposttraj_ev = get_vehicle_vtraj_per_id(data_ev, t_cur, id_ev, horizon_post, do_reverse=0, handle_remain=1)
    prevtraj_ev, _ = get_vehicle_traj_per_id(data_ev, t_cur, id_ev, horizon_prev, do_reverse=1, handle_remain=1)

    posttraj_ev_sp, prevtraj_ev_sp = posttraj_ev[idx_post_sp, :], prevtraj_ev[idx_prev_sp, :]

    posttraj_ev_enc = encode_traj(posttraj_ev, 1, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track)
    prevtraj_ev_enc = encode_traj(prevtraj_ev, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track)
    posttraj_ev_enc_sp = encode_traj(posttraj_ev_sp, 1, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track)
    prevtraj_ev_enc_sp = encode_traj(prevtraj_ev_sp, 0, sim_track.pnts_poly_track, sim_track.pnts_lr_border_track)

    posttraj_ovnear, vtraj_ovnear, prevtraj_ovnear = [], [], []
    posttraj_ovnear_sp, prevtraj_ovnear_sp = [], []
    s_ovnear = np.zeros((len(id_near), 4), dtype=np.float32)
    size_ovnear = np.zeros((len(id_near), 2), dtype=np.float32)
    prevtraj_ovnear_sp_enc = np.zeros((1, num_near, dim_pretraj_sp), dtype=np.float32)
    for nidx_near in range(0, len(id_near)):
        id_near_sel = id_near[nidx_near]
        if id_near_sel == -1:
            posttraj_ov_tmp = -1000 * np.ones((horizon_post + 1, 3), dtype=np.float32)
            prevtraj_ov_tmp = -1000 * np.ones((horizon_prev + 1, 3), dtype=np.float32)
            vtraj_ov_tmp = np.zeros((horizon_prev + 1,), dtype=np.float32)
            posttraj_ov_sp_tmp = posttraj_ov_tmp[idx_post_sp, :]
            prevtraj_ov_sp_tmp = prevtraj_ov_tmp[idx_prev_sp, :]
        else:
            posttraj_ov_tmp, size_tmp = get_vehicle_traj_per_id(data_ov, t_cur, id_near_sel, horizon_post,
                                                                do_reverse=0, handle_remain=1)
            vtraj_ov_tmp = get_vehicle_vtraj_per_id(data_ov, t_cur, id_near_sel, horizon_post, do_reverse=0,
                                                    handle_remain=1)
            prevtraj_ov_tmp, _ = get_vehicle_traj_per_id(data_ov, t_cur, id_near_sel, horizon_prev, do_reverse=1,
                                                         handle_remain=1)
            posttraj_ov_sp_tmp = posttraj_ov_tmp[idx_post_sp, :]
            prevtraj_ov_sp_tmp = prevtraj_ov_tmp[idx_prev_sp, :]

            prevtraj_ov_sp_enc_tmp = encode_traj(prevtraj_ov_sp_tmp, 0, sim_track.pnts_poly_track,
                                                 sim_track.pnts_lr_border_track)

            size_ovnear[nidx_near, :] = size_tmp
            prevtraj_ovnear_sp_enc[0, nidx_near, :] = prevtraj_ov_sp_enc_tmp

        s_ovnear[nidx_near, :] = [posttraj_ov_tmp[0, 0], posttraj_ov_tmp[0, 1], posttraj_ov_tmp[0, 2], vtraj_ov_tmp[0]]
        posttraj_ovnear.append(posttraj_ov_tmp)
        vtraj_ovnear.append(vtraj_ov_tmp)
        prevtraj_ovnear.append(prevtraj_ov_tmp)
        posttraj_ovnear_sp.append(posttraj_ov_sp_tmp)
        prevtraj_ovnear_sp.append(prevtraj_ov_sp_tmp)

    return f, id_near, lanewidth_cur, pnt_center, rad_center, pnt_left, pnt_right, posttraj_ev, \
           vposttraj_ev, posttraj_ev_sp, prevtraj_ev, posttraj_ev_enc, prevtraj_ev_enc, posttraj_ev_enc_sp, prevtraj_ev_enc_sp, \
           s_ovnear, posttraj_ovnear, vtraj_ovnear, prevtraj_ovnear, posttraj_ovnear_sp, prevtraj_ovnear_sp, \
           prevtraj_ovnear_sp_enc, size_ovnear


def get_current_info_part_near(id_near, data_v, sim_track):
    """
    Gets current info (part, near).
    data-vehicle: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    """

    num_near = len(id_near)

    id_near_ovnear = np.zeros((num_near, num_near), dtype=np.int32)
    lanewidth_ovnear = np.zeros((num_near,), dtype=np.float32)
    pnt_center_ovnear = np.zeros((num_near, 2), dtype=np.float32)
    rad_center_ovnear = np.zeros((num_near,), dtype=np.float32)
    pnt_left_ovnear = np.zeros((num_near, 2), dtype=np.float32)
    pnt_right_ovnear = np.zeros((num_near, 2), dtype=np.float32)

    for nidx_near in range(0, num_near):
        id_sel = id_near[nidx_near]

        if id_sel == -1:
            continue

        # Set data (tv, ov)
        idx_tv_ = np.where(data_v[:, -1] == id_sel)
        idx_tv = idx_tv_[0]
        data_tv = data_v[idx_tv, :]
        idx_ov = np.setdiff1d(np.arange(0, data_v.shape[0]), idx_tv)
        data_ov = data_v[idx_ov, :]

        # Get feature info
        f_out, id_near_out, pnts_debug_f_ev, _, rad_center, _ = get_feature(sim_track, data_tv, data_ov, use_intp=0)
        lanewidth = f_out[3]  # Current lanewidth
        pnt_center, pnt_left, pnt_right = pnts_debug_f_ev[0, :], pnts_debug_f_ev[1, :], pnts_debug_f_ev[2, :]

        id_near_ovnear[nidx_near, :] = id_near_out
        lanewidth_ovnear[nidx_near] = lanewidth
        pnt_center_ovnear[nidx_near, :] = pnt_center
        rad_center_ovnear[nidx_near] = rad_center
        pnt_left_ovnear[nidx_near, :] = pnt_left
        pnt_right_ovnear[nidx_near, :] = pnt_right

    return id_near_ovnear, lanewidth_ovnear, pnt_center_ovnear, rad_center_ovnear, pnt_left_ovnear, pnt_right_ovnear


def recover_trajectory_prediction(s_ev, val_y_pred, y_mean_train, y_scale_train, dim_p, h_y, yth, sim_track,
                                  is_track_simple=0):
    """
    Recovers trajectory prediction.
    :param val_y_pred: (ndarray) [batch_size=1, timesteps, dim_p] or [timesteps, dim_p]
    """
    if len(y_mean_train) == 0 or len(y_scale_train) == 0:
        do_recover = False
    else:
        do_recover = True

    val_y_pred = val_y_pred.reshape(1, -1)

    if do_recover:
        val_y_pred_rc = recover_from_normalized_data(val_y_pred, y_mean_train, y_scale_train)
    else:
        val_y_pred_rc = val_y_pred

    # Vehicle cannot go back in a horizontal direction to the lane
    size_tmp = val_y_pred_rc.shape[1]
    idx_1_tmp = np.arange(0, size_tmp, dim_p)
    idx_2_tmp = np.where(val_y_pred_rc[0, idx_1_tmp] < 0.0)
    idx_2_tmp = idx_2_tmp[0]
    if len(idx_2_tmp) > 0:
        idx_3_tmp = idx_1_tmp[idx_2_tmp]
        val_y_pred_rc[0, idx_3_tmp] = 0

    # Vehicle cannot shift vertically up to a certain level
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

    traj_decoded = decode_traj_2(s_ev[0:3], val_y_pred_rc, h_y, 1, sim_track, is_track_simple=is_track_simple,
                                 dim_p=dim_p)

    return traj_decoded
