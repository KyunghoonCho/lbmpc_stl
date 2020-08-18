# UTILITY-FUNCTIONS (STL-ROBUSTNESS)

from __future__ import print_function

import numpy as np
import math
import matplotlib.pyplot as plt

from src.utils import *
from src.utils_sim import *
from src.get_rgb import *
from matplotlib import cm

__all__ = ["compute_lane_constraints", "compute_collision_constraints", "get_modified_size_linear", "convert_state",
           "convert_trajectory", "compute_robustness_lane", "compute_robustness_collision", "compute_robustness_speed",
           "compute_robustness_until", "compute_robustness_part", "compute_robustness",
           "compute_robustness_from_data", "compute_robustness_from_traj",
           "scale_robustness", "plot_robustness_bar", "plot_robustness_bar_twin", "plot_robustness_history"]


def compute_lane_constraints(pnt_down, pnt_up, ry, lane_angle, margin_dist, cp2rotate, theta2rotate):
    """
    Computes lane constraints.
    :param pnt_down: (down) point (dim = 2)
    :param pnt_up: (up) point (dim = 2)
    :param ry: width-size (float)
    :param lane_angle: lane-heading (rad, float)
    :param margin_dist: margin dist (bigger -> loosing constraints, float)
    :param cp2rotate: center point to convert (dim = 2)
    :param theta2rotate: angle (rad) to convert (float)
    """
    pnt_down = make_numpy_array(pnt_down, keep_1dim=True)
    pnt_up = make_numpy_array(pnt_up, keep_1dim=True)

    pnt_down_r = np.reshape(pnt_down[0:2], (1, 2))
    pnt_up_r = np.reshape(pnt_up[0:2], (1, 2))
    pnt_down_conv_ = get_rotated_pnts_tr(pnt_down_r, -cp2rotate, -theta2rotate)
    pnt_up_conv_ = get_rotated_pnts_tr(pnt_up_r, -cp2rotate, -theta2rotate)
    pnt_down_conv, pnt_up_conv = pnt_down_conv_[0, :], pnt_up_conv_[0, :]
    lane_angle_r = angle_handle(lane_angle - theta2rotate)

    margin_dist = margin_dist - ry / 2.0
    cp_l_d = pnt_down_conv + np.array([+margin_dist*math.sin(lane_angle_r),
                                       -margin_dist*math.cos(lane_angle_r)], dtype=np.float32)
    cp_l_u = pnt_up_conv + np.array([-margin_dist * math.sin(lane_angle_r),
                                     +margin_dist * math.cos(lane_angle_r)], dtype=np.float32)
    rad_l = lane_angle_r

    return cp_l_d, cp_l_u, rad_l


def compute_collision_constraints(h, xinit_conv, ry, trajs_ov_near, sizes_ov_near, id_near, cp2rotate, theta2rotate):
    """
    Computes collision constraints.
    :param h: horizon (int)
    :param xinit_conv: init state (position-x, position-y)
    :param ry: width-size (float)
    :param trajs_ov_near: (list)-> (ndarray) x y theta (dim = N x 3), [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    :param sizes_ov_near: dx dy (dim = 2)
    :param id_near: selected indexes (dim = 6), [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
    :param cp2rotate: center point to convert (dim = 2)
    :param theta2rotate: angle (rad) to convert (float)
    """

    xinit_conv = make_numpy_array(xinit_conv, keep_1dim=True)
    id_near = make_numpy_array(id_near, keep_1dim=True)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    # Do reset
    traj_ov, size_ov = [], []
    traj_ov_cf, size_ov_cf = [], []

    len_near = len(trajs_ov_near)

    for nidx_l in range(0, len_near):
        traj_ov_near_list_sel = trajs_ov_near[nidx_l]
        size_ov_near_list_sel = sizes_ov_near[nidx_l]
        id_near_sel = id_near[nidx_l]

        traj_ov_near_list_sel = make_numpy_array(traj_ov_near_list_sel, keep_1dim=False)
        size_ov_near_list_sel = make_numpy_array(size_ov_near_list_sel, keep_1dim=True)

        if nidx_l == 4:  # id_cf
            if id_near_sel == -1:
                traj_ov_cf = traj_ov_near_list_sel
                size_ov_cf = np.zeros((h + 1, 2), dtype=np.float32)
            else:
                traj_tmp = np.zeros((h + 1, 3), dtype=np.float32)
                traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                diff_tmp = traj_tmp[:, 0:2] - np.reshape(xinit_conv[0:2], (1, 2))
                dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                idx_tmp_ = np.where(dist_tmp > 100.0)
                idx_tmp_ = idx_tmp_[0]
                if len(idx_tmp_) > 0:
                    traj_tmp[idx_tmp_, 0:2] = [xinit_conv[0] - 200, xinit_conv[1] - 200]

                traj_ov_cf = traj_tmp

                size_ov_cf = np.zeros((h + 1, 2), dtype=np.float32)
                for nidx_h in range(0, h + 1):
                    traj_tmp_sel = traj_tmp[nidx_h, :]
                    size_ov_near_list_sel_new = np.array([size_ov_near_list_sel[0], size_ov_near_list_sel[1]],
                                                         dtype=np.float32)
                    if size_ov_near_list_sel_new[1] < 0.9 * ry:
                        size_ov_near_list_sel_new[1] = 0.9 * ry
                    size_tmp_sel = get_modified_size_linear(size_ov_near_list_sel_new, traj_tmp_sel[2])
                    size_ov_cf[nidx_h, :] = size_tmp_sel

        else:
            if id_near_sel == -1:
                traj_tmp = traj_ov_near_list_sel
                size_tmp = np.zeros((h + 1, 2), dtype=np.float32)
            else:
                traj_tmp = np.zeros((h + 1, 3), dtype=np.float32)
                traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                diff_tmp = traj_tmp[:, 0:2] - np.reshape(xinit_conv[0:2], (1, 2))
                dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                idx_tmp_ = np.where(dist_tmp > 100.0)
                idx_tmp_ = idx_tmp_[0]
                if len(idx_tmp_) > 0:
                    traj_tmp[idx_tmp_, 0:2] = [xinit_conv[0] - 100, xinit_conv[1] - 100]

                size_tmp = np.zeros((h + 1, 2), dtype=np.float32)
                for nidx_h in range(0, h + 1):
                    traj_tmp_sel = traj_tmp[nidx_h, :]
                    size_tmp_sel = get_modified_size_linear(size_ov_near_list_sel, traj_tmp_sel[2])
                    size_tmp[nidx_h, :] = size_tmp_sel

            traj_ov.append(traj_tmp)
            size_ov.append(size_tmp)

    return traj_ov_cf, size_ov_cf, traj_ov, size_ov


def get_modified_size_linear(size, theta, w=0.4):
    """
    Gets modified size for linearization.
    :param size: dx dy (dim = 2)
    :param theta: heading (float)
    :param w: weight (float)
    """
    size = make_numpy_array(size, keep_1dim=True)

    if abs(math.cos(theta)) <= 0.125:
        size_out = np.flipud(size)
    elif abs(math.sin(theta)) <= 0.125:
        size_out = size
    elif abs(math.cos(theta)) < 1 / math.sqrt(2):
        size_x = w * abs(size[0] * math.cos(theta)) + abs(size[1] * math.sin(theta))
        size_y = abs(size[0] * math.sin(theta)) + w * abs(size[1] * math.cos(theta))
        size_out = np.array([size_x, size_y], dtype=np.float32)
    else:
        size_x = abs(size[0] * math.cos(theta)) + w * abs(size[1] * math.sin(theta))
        size_y = w * abs(size[0] * math.sin(theta)) + abs(size[1] * math.cos(theta))
        size_out = np.array([size_x, size_y], dtype=np.float32)

    return size_out


def convert_state(x, cp2rotate, theta2rotate):
    """
    Converts state (rotates w.r.t. reference pose).
    :param x: vehicle system state (dim = 4)
    :param cp2rotate: center point to convert (dim = 2)
    :param theta2rotate: angle (rad) to convert (float)
    """
    x = make_numpy_array(x, keep_1dim=True)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    dim_x = 4

    # Convert state
    xconv_in = np.zeros((dim_x,), dtype=np.float32)
    x_in_r = np.reshape(x[0:2], (1, 2))
    xconv_in_tmp1 = get_rotated_pnts_tr(x_in_r, -cp2rotate, -theta2rotate)
    # xconv_in_tmp1 = get_rotated_pnts_rt(x_in_r, -cp2rotate, -theta2rotate)
    xconv_in_tmp1 = xconv_in_tmp1.reshape(-1)
    xconv_in[0:2] = xconv_in_tmp1
    xconv_in[2] = angle_handle(x[2] - theta2rotate)
    xconv_in[3] = x[3]

    return xconv_in


def convert_trajectory(traj, cp2rotate, theta2rotate):
    """
    Converts trajectory.
    :param traj: trajectory (dim = N x 4 or N x 3)
    :param cp2rotate: center point to convert (dim = 2)
    :param theta2rotate: angle (rad) to convert (float)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    cp2rotate = make_numpy_array(cp2rotate, keep_1dim=True)

    len_traj = traj.shape[0]
    dim_x = traj.shape[1]

    traj_conv = np.zeros((len_traj, dim_x), dtype=np.float32)
    traj_conv[:, 0:2] = get_rotated_pnts_tr(traj[:, 0:2], -cp2rotate, -theta2rotate)
    traj_conv[:, 2] = angle_handle(traj[:, 2] - theta2rotate)
    if dim_x == 4:
        traj_conv[:, 3] = traj[:, 3]

    return traj_conv


def compute_robustness_lane(traj, cp_l_d, cp_l_u, rad_l):
    """
    Computes robustness (lane-observation: down & up).
    :param traj: trajectory (dim = N x 2)
    :param cp_l_d: (down) point (dim = 2)
    :param cp_l_u: (up) point (dim = 2)
    :param rad_l: lane-angle (rad, float)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    cp_l_d = make_numpy_array(cp_l_d, keep_1dim=True)
    cp_l_u = make_numpy_array(cp_l_u, keep_1dim=True)

    h = traj.shape[0]

    r_l_down_array = np.zeros((h,), dtype=np.float32)
    r_l_up_array = np.zeros((h,), dtype=np.float32)
    for nidx_h in range(0, h):
        y_down = math.tan(rad_l) * (traj[nidx_h, 0] - cp_l_d[0]) + cp_l_d[1]
        r_l_down_tmp = traj[nidx_h, 1] - y_down

        y_up = math.tan(rad_l) * (traj[nidx_h, 0] - cp_l_u[0]) + cp_l_u[1]
        r_l_up_tmp = y_up - traj[nidx_h, 1]

        r_l_down_array[nidx_h] = r_l_down_tmp
        r_l_up_array[nidx_h] = r_l_up_tmp

    r_l_down = np.min(r_l_down_array)
    r_l_up = np.min(r_l_up_array)

    return r_l_down, r_l_up, r_l_down_array, r_l_up_array


def compute_robustness_collision(traj, size, traj_oc, size_oc, param_x_1=1.0, param_x_2=0.125,
                                 param_x_3=1.25, param_y_1=0.1, param_y_2=0.125, param_y_3=0.1, param_y_4=0.2,
                                 lanewidth=3.28, use_mod=1):
    """
    Computes robustness (collision).
    :param traj: trajectory (dim = N x 2)
    :param size: dx dy (dim = 2)
    :param traj_oc: other-vehicle trajectory (dim = N x 2)
    :param size_oc: other-vehicle size (dim = N x 2)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    size = make_numpy_array(size, keep_1dim=True)
    traj_oc = make_numpy_array(traj_oc, keep_1dim=False)
    size_oc = make_numpy_array(size_oc, keep_1dim=False)

    h = traj.shape[0]

    if h > 1:
        diff_traj_tmp = traj[range(1, h), 0:2] - traj[range(0, (h - 1)), 0:2]
        dist_traj_tmp = np.sqrt(np.sum(diff_traj_tmp * diff_traj_tmp, axis=1))

    r_oc_array = np.zeros((h,), dtype=np.float32)
    dist_traj = 0.0
    for nidx_h in range(0, h):
        x_oc_tmp, y_oc_tmp = traj_oc[nidx_h, 0], traj_oc[nidx_h, 1]

        if use_mod == 1:
            # Modify size w.r.t. distance
            if h > 1:
                if nidx_h < dist_traj_tmp.shape[0]:
                    dist_traj = min(dist_traj + dist_traj_tmp[nidx_h] * lanewidth / 3.28, 1000)
            mod_size_x = min(param_x_1 * (math.exp(param_x_2 * dist_traj) - 1.0), param_x_3)
            mod_size_y = param_y_1 + min(param_y_2 * (math.exp(param_y_3 * dist_traj) - 1.0), param_y_4)
        else:
            mod_size_x, mod_size_y = 0.0, 0.0
        rx_oc = (size_oc[nidx_h, 0] + size[0]) / 2.0 + mod_size_x
        ry_oc = (size_oc[nidx_h, 1] + size[1]) / 2.0 + mod_size_y

        r_b_oc0, r_b_oc1 = float(-x_oc_tmp - rx_oc), float(x_oc_tmp - rx_oc)
        r_b_oc2, r_b_oc3 = float(-y_oc_tmp - ry_oc), float(y_oc_tmp - ry_oc)

        r_oc0, r_oc1 = traj[nidx_h, 0] + r_b_oc0, -traj[nidx_h, 0] + r_b_oc1
        r_oc2, r_oc3 = traj[nidx_h, 1] + r_b_oc2, -traj[nidx_h, 1] + r_b_oc3

        r_oc_tmp = max([r_oc0, r_oc1, r_oc2, r_oc3])
        r_oc_array[nidx_h] = r_oc_tmp

    r_oc = np.min(r_oc_array)

    return r_oc, r_oc_array


def compute_robustness_speed(vtraj, v_th):
    """
    Computes robustness (speed).
    :param vtraj: velocity trajectory (dim = N)
    :param v_th: threshold velocity (float)
    """
    vtraj = make_numpy_array(vtraj, keep_1dim=True)

    len_v_traj_in = vtraj.shape[0]
    r_speed_array = np.zeros((len_v_traj_in, ), dtype=np.float32)
    for nidx_h in range(0, len_v_traj_in):
        r_speed_array[nidx_h] = v_th - vtraj[nidx_h]

    r_speed = np.min(r_speed_array)

    return r_speed, r_speed_array


def compute_robustness_until(traj, size, vtraj, traj_cf, size_cf, t_s, t_a, t_b, v_th, d_th):
    """
    Computes robustness (until).
    :param traj: trajectory (dim = N x 2)
    :param size: dx dy (dim = 2)
    :param vtraj: velocity trajectory (dim = N)
    :param traj_cf: other-vehicle (center-front) trajectory (dim = N x 2)
    :param size_cf: other-vehicle (center-front) size (dim = N x 2)
    :param t_s: until-logic time (int)
    :param t_a: until-logic time (int)
    :param t_b: until-logic time (int)
    :param v_th: until-logic velocity threshold (float)
    :param d_th: until-logic distance threshold (float)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    size = make_numpy_array(size, keep_1dim=True)
    vtraj = make_numpy_array(vtraj, keep_1dim=True)
    traj_cf = make_numpy_array(traj_cf, keep_1dim=False)
    size_cf = make_numpy_array(size_cf, keep_1dim=False)

    len_1 = t_b - t_a + 1
    r_1 = np.zeros((len_1, ), dtype=np.float32)
    for idx_t1 in range(t_a, t_b + 1):
        r_phi_2 = traj[idx_t1, 0] - traj_cf[idx_t1, 0] + (size[0] + size_cf[idx_t1, 0]) / 2.0 + d_th

        len_3_tmp = idx_t1 - t_s + 1
        r_3_tmp = np.zeros((len_3_tmp, ), dtype=np.float32)
        for idx_t2 in range(t_s, idx_t1 + 1):
            r_phi_1 = v_th - vtraj[idx_t2]
            r_3_tmp[idx_t2 - t_s] = r_phi_1

        rmin_3_tmp = np.amin(r_3_tmp, axis=0)
        r_1[idx_t1 - t_a] = np.amin(np.array([r_phi_2, rmin_3_tmp]), axis=0)

    r_out = np.amax(r_1)
    return r_out


def compute_robustness_part(traj, size, cp_l_d, cp_l_u, rad_l, traj_cf, size_cf, traj_rest, size_rest, idx_h_ov, v_th):
    """
    Computes robustness (part).
        1: Lane-observation (down, right)
        2: Lane-observation (up, left)
        3: Collision (front)
        4: Collision (others)
        5: Speed-limit
    :param traj: trajectory ([x, y, theta, v], dim = N x 4)
    :param size: dx dy (dim = 2)
    :param cp_l_d: (down) point (dim = 2)
    :param cp_l_u: (up) point (dim = 2)
    :param rad_l: lane-angle (rad, float)
    :param traj_cf: other-vehicle (center-front) trajectory (dim = N x 2)
    :param size_cf: other-vehicle (center-front) size (dim = N x 2)
    :param traj_rest: other-vehicle (rest) trajectory (dim = N x 2)
    :param size_rest: other-vehicle (rest) size (dim = N x 2)
    :param idx_h_ov: indexes to compute (other-vehicle)
    :param v_th: velocity threshold (float)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    size = make_numpy_array(size, keep_1dim=True)

    # Rule 1-2: lane-observation (down, up)
    r_l_down, r_l_up, r_l_down_array, r_l_up_array = compute_robustness_lane(traj, cp_l_d, cp_l_u, rad_l)

    # Rule 3: collision (front)
    r_c_cf, r_c_cf_array = compute_robustness_collision(traj, size, traj_cf[idx_h_ov, :], size_cf[idx_h_ov, :],
                                                        use_mod=0)

    # Rule 4: collision (rest)
    num_oc_rest = len(traj_rest)
    r_oc_rest_, r_oc_rest_array_ = [], []
    for nidx_oc in range(0, num_oc_rest):
        traj_rest_in_tmp = traj_rest[nidx_oc]
        size_rest_in_tmp = size_rest[nidx_oc]
        r_c_tmp, r_c_array_tmp = compute_robustness_collision(traj, size, traj_rest_in_tmp[idx_h_ov, :],
                                                              size_rest_in_tmp[idx_h_ov, :])
        r_oc_rest_.append(r_c_tmp)
        r_oc_rest_array_.append(r_c_array_tmp)
    r_oc_rest_ = np.asarray(r_oc_rest_)

    r_c_rest = np.min(r_oc_rest_)
    r_c_rest_array = r_oc_rest_

    # Rule 5: speed-limit
    r_speed, r_speed_array = compute_robustness_speed(traj[:, -1], v_th)

    return r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed


def compute_robustness(traj, vtraj, size, cp_l_d, cp_l_u, rad_l, traj_cf, size_cf, traj_rest,
                       size_rest, idx_h_ov, v_th, until_t_s, until_t_a, until_t_b, until_v_th, until_d_th):
    """
    Computes robustness.
        1: Lane-observation (down, right)
        2: Lane-observation (up, left)
        3: Collision (front)
        4: Collision (others)
        5: Speed-limit
        6: Slow-down
    :param traj: trajectory ([x, y], dim = N x 2)
    :param vtraj: vehicle trajectory (dim = N)
    :param size: dx dy (dim = 2)
    :param cp_l_d: (down) point (dim = 2)
    :param cp_l_u: (up) point (dim = 2)
    :param rad_l: lane-angle (rad, float)
    :param traj_cf: other-vehicle (center-front) trajectory (dim = N x 2)
    :param size_cf: other-vehicle (center-front) size (dim = N x 2)
    :param traj_rest: other-vehicle (rest) trajectory (dim = N x 2)
    :param size_rest: other-vehicle (rest) size (dim = N x 2)
    :param idx_h_ov: indexes to compute (other-vehicle)
    :param v_th: velocity threshold (float)
    :param until_t_s: until-logic time (int)
    :param until_t_a: until-logic time (int)
    :param until_t_b: until-logic time (int)
    :param until_v_th: until-logic velocity threshold (float)
    :param until_d_th: until-logic distance threshold (float)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    vtraj = make_numpy_array(vtraj, keep_1dim=True)
    size = make_numpy_array(size, keep_1dim=True)

    traj_concat = np.concatenate((traj, vtraj.reshape((-1, 1))), axis=1)

    # Rule 1-2: lane-observation (down, up)
    # Rule 3: collision (front)
    # Rule 4: collision (rest)
    # Rule 5: speed-limit
    r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed = \
        compute_robustness_part(traj_concat, size, cp_l_d, cp_l_u, rad_l, traj_cf, size_cf, traj_rest, size_rest,
                                idx_h_ov, v_th)

    # Rule 6: until-logic
    r_until = compute_robustness_until(traj[:, 0:2], size, vtraj, traj_cf[idx_h_ov, :],
                                       size_cf[idx_h_ov, :], until_t_s, until_t_a, until_t_b, until_v_th, until_d_th)

    return r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed, r_until


def compute_robustness_from_data(id_tv, t_cur, data_v, y_tv, vy_tv, h_y, id_near, lanewidth, pnt_center, rad_center,
                                 pnt_left, pnt_right, v_th, until_t_s, until_t_a, until_t_b, until_v_th, until_d_th,
                                 until_lanewidth):
    """
    Computes robustness from vehicle data.
        structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    """
    # Set data (tv, ov)
    data_tv, data_ov = select_data_ids(data_v, [id_tv])
    data_tv_cur, _ = select_data_t(data_tv, t_cur, keep_1dim=True)
    s_tv_cur = np.array([data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[4]], dtype=np.float32)
    rx_tv, ry_tv, = data_tv_cur[6], data_tv_cur[5]

    # Get target vehicle trajectory
    if len(y_tv) == 0:
        y_tv, _ = get_vehicle_traj_per_id(data_tv, t_cur, id_tv, h_y, do_reverse=0, handle_remain=3)
    if len(vy_tv) == 0:
        vy_tv = get_vehicle_vtraj_per_id(data_tv, t_cur, id_tv, h_y, do_reverse=0, handle_remain=1)

    # Get near other vehicles trajectory
    data_ov_near, data_ov_near_list = select_data_near(data_ov, id_near)
    y_near_list, ysize_near_list = get_vehicle_traj_near(data_ov_near_list, id_near, t_cur, h_y, do_reverse=0,
                                                         handle_remain=3)

    cp2rotate_mpc, theta2rotate_mpc = pnt_center, rad_center
    s_tv_cur_conv = convert_state(s_tv_cur, cp2rotate_mpc, theta2rotate_mpc)
    y_tv_conv = convert_trajectory(y_tv, cp2rotate_mpc, theta2rotate_mpc)
    cp_l_d, cp_l_u, rad_l = compute_lane_constraints(pnt_right, pnt_left, ry_tv, rad_center, 0.0, cp2rotate_mpc,
                                                     theta2rotate_mpc)
    y_cf, ysize_cf, y_ov, ysize_ov = compute_collision_constraints(h_y, s_tv_cur_conv, ry_tv, y_near_list,
                                                                   ysize_near_list, id_near, cp2rotate_mpc,
                                                                   theta2rotate_mpc)

    idx_h_tmp, idx_h_ov_tmp = np.arange(1, h_y + 1), np.arange(1, h_y + 1)
    until_v_th_mod = until_v_th * lanewidth / until_lanewidth
    until_d_th_mod = until_d_th * lanewidth / until_lanewidth
    r_l_down, r_l_up, r_c_cf, r_c_rest, _, r_speed, r_until = compute_robustness(
        y_tv_conv[idx_h_tmp, :], vy_tv[idx_h_tmp], [rx_tv, ry_tv], cp_l_d, cp_l_u, rad_l, y_cf, ysize_cf, y_ov,
        ysize_ov, idx_h_ov_tmp, v_th, until_t_s, until_t_a, until_t_b, until_v_th_mod, until_d_th_mod)

    robust_tv_gt = np.array([r_l_down, r_l_up, r_c_cf, r_c_rest, r_speed, r_until], dtype=np.float32)

    return robust_tv_gt, cp_l_d, cp_l_u, rad_l


def compute_robustness_from_traj(id_ev, t_cur, data_v, y_ev, y_ov_near, h_y, id_near, lanewidth, pnt_center, rad_center,
                                 pnt_left, pnt_right, dt, v_th, until_t_s, until_t_a, until_t_b, until_v_th, until_d_th,
                                 until_lanewidth):
    """
    Computes robustness from trajectory.
    """
    # Set data (ev, ov)
    data_ev, data_ov = select_data_ids(data_v, [id_ev])
    data_ev_cur, _ = select_data_t(data_ev, t_cur, keep_1dim=True)
    s_ev_cur = np.array([data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[4]], dtype=np.float32)
    rx_ev, ry_ev, = data_ev_cur[6], data_ev_cur[5]

    # Get 'vy_ev'
    vy_ev = np.zeros((h_y + 1,), dtype=np.float32)
    y_ev_tmp = y_ev[:, 0:2]
    diff_tmp = y_ev_tmp[0:h_y, :] - y_ev_tmp[1:(h_y + 1), :]
    dist_tmp = np.sqrt(np.sum(diff_tmp*diff_tmp, 1))
    vdist_tmp = dist_tmp / dt
    vy_ev[0:h_y] = vdist_tmp
    vy_ev[-1] = vy_ev[-2]

    # Get 'size' of near-vehicles
    size_ov_near_list = get_vehicle_size(data_v, id_near)

    # Compute robustness
    cp2rotate_mpc, theta2rotate_mpc = pnt_center, rad_center
    s_ev_cur_conv = convert_state(s_ev_cur, cp2rotate_mpc, theta2rotate_mpc)
    y_ev_conv = convert_trajectory(y_ev, cp2rotate_mpc, theta2rotate_mpc)
    cp_l_d, cp_l_u, rad_l = compute_lane_constraints(pnt_right, pnt_left, ry_ev, rad_center, 0.0, cp2rotate_mpc,
                                                     theta2rotate_mpc)
    y_ov_cf, size_ov_cf, y_ov_rest, size_ov_rest = \
        compute_collision_constraints(h_y, s_ev_cur_conv, ry_ev, y_ov_near, size_ov_near_list, id_near, cp2rotate_mpc,
                                      theta2rotate_mpc)

    idx_h_sel_tmp, idx_h_ov_tmp = np.arange(1, h_y + 1), np.arange(1, h_y + 1)
    until_v_th_mod = until_v_th * lanewidth / until_lanewidth
    until_d_th_mod = until_d_th * lanewidth / until_lanewidth
    r_l_down, r_l_up, r_c_cf, r_c_rest, _, r_speed, r_until = compute_robustness(
        y_ev_conv[idx_h_sel_tmp, :], vy_ev[idx_h_sel_tmp], [rx_ev, ry_ev], cp_l_d, cp_l_u,
        rad_l, y_ov_cf, size_ov_cf, y_ov_rest, size_ov_rest, idx_h_ov_tmp, v_th, until_t_s, until_t_a, until_t_b,
        until_v_th_mod, until_d_th_mod)

    robust_ev = np.array([r_l_down, r_l_up, r_c_cf, r_speed, r_until], dtype=np.float32)

    return robust_ev


def scale_robustness(y, y_absmax):
    """
    Scales robustness between -1 ~ +1.
    :param y: vector-in (dim = N)
    :param y_absmax: abs-max (dim = N)
    """
    # y, y_scale: (ndarray or list)

    y = make_numpy_array(y, keep_1dim=True)

    y_out = np.copy(y)
    for nidx_y in range(0, y_out.shape[0]):
        y_out[nidx_y] = np.maximum(np.minimum(y_out[nidx_y], y_absmax[nidx_y]), -y_absmax[nidx_y]) / y_absmax[nidx_y]

    return y_out


# PLOT ----------------------------------------------------------------------------------------------------------------#
def plot_robustness_bar(r, bar_width, do_save, filename2save, hcolor1, hcolor2, is_background_black=1):
    """
    Plots robustness (bar).
    r: value (0 ~ 1)
    bar_width: 0 ~ 1 (default: 0.2)
    """

    fig = plt.figure()
    if is_background_black == 1:
        fig.patch.set_facecolor('black')
        edgecolor = 'w'
    else:
        edgecolor = 'k'

    ny = len(r)
    ind = np.arange(ny)
    barlist = plt.bar(ind, r, bar_width, edgecolor=edgecolor, linewidth=1.2)

    # Set color
    for nidx_r in range(0, len(barlist)):
        r_sel = r[nidx_r]

        alpha = max(min((r_sel + 1) / 2.0, 1), 0)
        hcolor_tmp = get_color_mix(alpha, hcolor1, hcolor2)

        barlist[nidx_r].set_facecolor(hcolor_tmp)

    plt.ylim([-1.1, +1.1])
    plt.axis('off')
    plt.show(block=False)

    if do_save >= 1:
        plt.savefig(filename2save, facecolor=fig.get_facecolor(), transparent=True, dpi=200)
    plt.pause(0.1)
    plt.close()


def plot_robustness_bar_twin(r0, r1, bar_width, do_save, filename2save, hcolor1, hcolor2, is_background_black=1):
    """
    Plots robustness (bar, twin).
    r0, r1: value (0 ~ 1)
    bar_width: 0 ~ 1 (default: 0.2)
    """

    fig, ax = plt.subplots()
    if is_background_black == 1:
        fig.patch.set_facecolor('black')
        edgecolor = 'w'
    else:
        edgecolor = 'k'

    ny = len(r0)
    ind = np.arange(ny)
    barlist1 = plt.bar(ind, r0, bar_width, edgecolor=edgecolor, linewidth=1.2)
    barlist2 = plt.bar(ind + bar_width, r1, bar_width, edgecolor=edgecolor, linewidth=1.2)

    # Set color
    for nidx_r in range(0, len(barlist1)):
        r0_sel = r0[nidx_r]
        r1_sel = r1[nidx_r]

        alpha_0 = max(min((r0_sel + 1) / 2.0, 1), 0)
        alpha_1 = max(min((r1_sel + 1) / 2.0, 1), 0)

        hcolor_tmp0 = get_color_mix(alpha_0, hcolor1, hcolor2)
        hcolor_tmp1 = get_color_mix(alpha_1, hcolor1, hcolor2)

        barlist1[nidx_r].set_facecolor(hcolor_tmp0)
        barlist2[nidx_r].set_facecolor(hcolor_tmp1)

    plt.ylim([-1.1, +1.1])
    plt.axis('off')
    plt.show(block=False)
    ax.set_xticks(ind + bar_width / 2)
    # ax.set_xticklabels(('{\varphi}_{1}', '{\varphi}_{2}', '{\varphi}_{3}', '{\varphi}_{4}', '{\varphi}_{5}'))
    # ax.legend((p1[0], p2[0]), ('Ground-truth', 'Prediction'))

    if do_save >= 1:
        plt.savefig(filename2save, facecolor=fig.get_facecolor(), transparent=True, dpi=200)
    plt.pause(0.1)
    plt.close()


def plot_robustness_history(y, plot_window, do_save, filename2save, is_background_black=1):
    """
    Plots robustness (history).
    """

    y_upper = np.ma.masked_where(y < 0, y)
    y_lower = np.ma.masked_where(y > 0, y)

    if is_background_black == 1:
        # fig.patch.set_facecolor('black')
        plt.style.use("dark_background")
        color_upper = get_rgb("Cyan")
        color_lower = get_rgb("Orange")
    else:
        plt.style.use("default")
        color_upper = get_rgb("Blue")
        color_lower = get_rgb("Red")

    dim_y, len_y = y.shape[1], y.shape[0]
    y_max, y_min = +1.2, -1.2

    major_ticks_x = np.arange(0, plot_window, plot_window / 4)
    minor_ticks_x = np.arange(0, plot_window, plot_window / 8)

    major_ticks_y = np.arange(y_min, y_max, (y_max - y_min) / 4)
    minor_ticks_y = np.arange(y_min, y_max, (y_max - y_min) / 8)

    fig, axs = plt.subplots(1, dim_y, figsize=(2 * dim_y + 1, 2), dpi=200)
    for nidx_y in range(0, dim_y):
        # axs[nidx_y].plot(y[:, nidx_y], "-", linewidth=1.5, color=color_upper, markevery=100)
        # axs[nidx_y].plot(len_y - 1, y[len_y - 1, nidx_y], marker="o", markersize=3, color=color_upper)

        axs[nidx_y].plot(y[:, nidx_y], "-", linewidth=1.5, color=get_rgb('Gray'), markevery=100)
        axs[nidx_y].plot(y_upper[:, nidx_y], "-", linewidth=1.5, color=color_upper, markevery=100)
        axs[nidx_y].plot(y_lower[:, nidx_y], "-", linewidth=1.5, color=color_lower, markevery=100)

        if y[len_y - 1, nidx_y] >= 0:
            axs[nidx_y].plot(len_y - 1, y[len_y - 1, nidx_y], marker="o", markersize=3, color=color_upper)
        else:
            axs[nidx_y].plot(len_y - 1, y[len_y - 1, nidx_y], marker="o", markersize=3, color=color_lower)

        # Set limits
        axs[nidx_y].set_xlim([0, plot_window])
        axs[nidx_y].set_ylim([y_min, y_max])

        # Set ticks
        axs[nidx_y].set_xticks(major_ticks_x)
        axs[nidx_y].set_xticks(minor_ticks_x, minor=True)
        axs[nidx_y].set_yticks(major_ticks_y)
        axs[nidx_y].set_yticks(minor_ticks_y, minor=True)

        # Set others
        axs[nidx_y].xaxis.set_ticklabels([])
        axs[nidx_y].yaxis.set_ticklabels([])
        axs[nidx_y].grid(which='major', alpha=0.5)
        axs[nidx_y].grid(which='minor', alpha=0.2)

    plt.show(block=False)

    if do_save >= 1:
        plt.savefig(filename2save, facecolor=fig.get_facecolor(), transparent=True, bbox_inches='tight', format='png')

    plt.pause(0.1)
    plt.close(fig)
