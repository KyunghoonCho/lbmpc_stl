from __future__ import print_function

import math
import numpy as np

from src.utils import *
from src.utils_sim import *


class SimControl(object):
    """
    EGO-VEHICLE FOR SIMULATOR
        System Dynamic: 4-state unicycle dynamics
        state: [x, y, theta, v]
        control: [w, a]
        dot(x) = v * cos(theta)
        dot(y) = v * sin(theta)
        dot(theta) = v * kappa_1 * w
        dot(v) = kappa_2 * a
    """
    def __init__(self, sim_track, dt_in, rx_in, ry_in, kappa, v_ref, v_range):
        self.dim_x, self.dim_u = 4, 2  # Set dimension for state & control

        self.track = sim_track  # track (class)

        self.dt = dt_in
        self.rx, self.ry = rx_in, ry_in
        self.kappa = kappa
        self.v_range = v_range  # [min, max]
        self.v_ref = v_ref

        # (Initial) Ego-state
        self.x_ego_init, self.y_ego_init, self.theta_ego_init, self.v_ego_init = 0.0, 0.0, 0.0, v_ref

        # Ego state
        self.x_ego, self.y_ego, self.theta_ego, self.v_ego = 0.0, 0.0, 0.0, v_ref

        # Ego control
        self.w_ego, self.a_ego = 0.0, 0.0

        # Initial points
        self.pnts_init = []

        # Reward components
        self.r_weight = []
        self.dev_rad_th, self.dist_cf_th, self.w_max = 0.0, 0.0, 0.0

    def set_reward_components(self, r_weight, dev_rad_th, dist_cf_th, w_max):
        """
        Set reward components.
        :param r_weight: reward weight
        :param dev_rad_th: heading threshold (float)
        :param dist_cf_th: cf distance threshold (float)
        :param w_max: max angular velocity (float)
        """
        # Reward weights
        # outside, collision, dev_rad, dev_dist, dist_cf, linear_velocity, angular_velocity, reach_goal (dim = 8)
        self.r_weight = r_weight

        self.dev_rad_th, self.dist_cf_th, self.w_max = dev_rad_th, dist_cf_th, w_max

    def set_track(self, sim_track):
        """
        Sets track.
        :param sim_track: track-info
        """
        self.track = sim_track

    def set_initstate(self, data_ov, t, segidx, laneidx, margin_rx, margin_ry, vel_rand=False):
        """
        Sets intial states.
        :param data_ov: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param t: time (float)
        :param segidx: segment index (int)
        :param laneidx: lane index (int)
        :param margin_rx: margin-x (float)
        :param margin_ry: margin-y (float)
        :param vel_rand: whether to set random velocity (boolean)
        """
        succeed_init, x_init, y_init, theta_init, pnts_init = \
            set_initstate(data_ov, t, self.rx, self.ry, segidx, laneidx, margin_rx, margin_ry, self.track)

        # Update
        if succeed_init == 1:
            self.pnts_init = pnts_init

            if vel_rand:
                v_init = self.v_ref + (np.random.random() - 0.5) * self.v_ref
            else:
                v_init = self.v_ref
            v_init = max(v_init, 0.1)

            self.update_initstate(x_init, y_init, theta_init, v_init)
            self.update_state(x_init, y_init, theta_init, v_init)
        else:
            print("[ERROR] FAILED TO INITSTATE!")

    def update_initstate(self, x, y, theta, v):
        """
        Updates init state.
        :param x: position-x (float)
        :param y: position-y (float)
        :param theta: heading (rad) (float)
        :param v: linear-velocity (float)
        """
        self.x_ego_init, self.y_ego_init, self.theta_ego_init, self.v_ego_init = x, y, theta, v

    def update_state(self, x, y, theta, v):
        """
        Updates state.
        :param x: position-x (float)
        :param y: position-y (float)
        :param theta: heading (rad) (float)
        :param v: linear-velocity (float)
        """
        self.x_ego, self.y_ego, self.theta_ego, self.v_ego = x, y, theta, v

    def update_control(self, w, a):
        """
        Updates control.
        :param w: angular-velocity (float)
        :param a: acceleration (float)
        """
        self.w_ego, self.a_ego = w, a

    def get_next_state(self, s_cur, w, a):
        """
        Gets next state.
        :param s_cur: x, y, theta, v (dim = 4)
        :param w: angular-velocity (float)
        :param a: acceleration (float)
        Dynamic
            dot(x) = v * cos(theta)
            dot(y) = v * sin(theta)
            dot(theta) = v * kappa_1 * av
            dot(v) = kappa_2 * a
        """
        x_new = s_cur[0] + math.cos(s_cur[2]) * s_cur[3] * self.dt
        y_new = s_cur[1] + math.sin(s_cur[2]) * s_cur[3] * self.dt
        theta_new = s_cur[2] + s_cur[3] * self.kappa[0] * w * self.dt
        lv_new = s_cur[3] + self.kappa[1] * a * self.dt

        theta_new = angle_handle(theta_new)
        theta_new = theta_new[0]

        s_new = np.array([x_new, y_new, theta_new, lv_new], dtype=np.float64)

        return s_new

    def get_traj_naive(self, u, horizon):
        """
        Gets (naive) trajectory.
        :param u: angular-velocity, acceleration (dim = 2)
        :param horizon: trajectory horizon (int)
        """
        traj = np.zeros((horizon + 1, 4), dtype=np.float64)
        traj[0, :] = [self.x_ego, self.y_ego, self.theta_ego, self.v_ego]

        for nidx_h in range(1, horizon + 1):
            s_prev = traj[nidx_h - 1, :]
            s_new = self.get_next_state(s_prev, u[0], u[1])
            traj[nidx_h, :] = s_new

        traj = traj.astype(dtype=np.float64)
        return traj

    def get_trajs_naive(self, u_set, horizon):
        """
        Get (naive) trajectory list.
        :param u_set: set of controls (ndarray, dim = N x 2)
        :param horizon: trajectory-horizon (int)
        """
        traj_list = []
        for nidx_d in range(0, u_set.shape[0]):
            u_sel = u_set[nidx_d, :]
            traj_out = self.get_traj_naive(u_sel, horizon)
            traj_list.append(traj_out)

        return traj_list

    def get_pnt_ahead(self, pos, dist_forward, segidx, laneidx):
        """
        Gets point ahead.
        :param pos: x, y (ndarray, dim = 2)
        :param dist_forward: distance forward (float)
        :param segidx: segment index (int)
        :param laneidx: lane index (int)
        """
        if segidx == -1 or laneidx == -1:
            seg_, lane_ = get_index_seglane(pos, self.track.pnts_poly_track)
            segidx, laneidx = seg_[0], lane_[0]

        pnts_c_tmp = self.track.pnts_m_track[segidx][laneidx]  # [0, :] start --> [end, :] end
        pnts_c_tmp = interpolate_data(pnts_c_tmp, data_type=0, alpha=10)

        if segidx < (self.track.num_seg - 1):
            segidx_next = segidx + 1
        else:
            segidx_next = 0 if self.track.is_circular else -1

        if segidx_next > -1:
            child_tmp = self.track.idx_child[segidx][laneidx]
            for nidx_tmp in range(0, len(child_tmp)):
                pnts_c_next_tmp = self.track.pnts_m_track[segidx_next][self.track.idx_child[segidx][laneidx][nidx_tmp]]
                # [0, :] start --> [end, :] end
                pnts_c_tmp = np.concatenate((pnts_c_tmp, pnts_c_next_tmp), axis=0)
            pnts_c = pnts_c_tmp
        else:
            pnts_c = pnts_c_tmp

        pos_r = np.reshape(pos, (1, 2))
        diff_tmp = np.tile(pos_r, (pnts_c.shape[0], 1)) - pnts_c[:, 0:2]
        dist_tmp = np.sqrt(np.sum(diff_tmp*diff_tmp, axis=1))
        idx_cur = np.argmin(dist_tmp, axis=0)

        dist_sum = 0.0
        idx_ahead = pnts_c.shape[0] - 1
        for nidx_d in range(idx_cur + 1, pnts_c.shape[0]):
            dist_c_tmp1 = (pnts_c[nidx_d, 0] - pnts_c[nidx_d - 1, 0]) * (pnts_c[nidx_d, 0] - pnts_c[nidx_d - 1, 0]) +\
                          (pnts_c[nidx_d, 1] - pnts_c[nidx_d - 1, 1]) * (pnts_c[nidx_d, 1] - pnts_c[nidx_d - 1, 1])
            dist_c_tmp2 = math.sqrt(dist_c_tmp1)
            dist_sum = dist_sum + dist_c_tmp2
            if dist_sum > dist_forward:
                idx_ahead = nidx_d
                break

        pnt_ahead = pnts_c[idx_ahead, :]
        pnt_ahead = pnt_ahead.reshape(-1)

        return pnt_ahead

    # SET GOAL --------------------------------------------------------------------------------------------------------#
    def get_goal_laneidx(self, cond_r, cond_l, segidx, laneidx):
        """
        Gets goal lane-index.
        :param cond_r: condition on move-right (boolean)
        :param cond_l: condition on move-left (boolean)
        :param segidx: current segment-index (int)
        :param laneidx: current lane-index (int)
        """
        lanedir = self.track.lane_dir[segidx][laneidx]

        if cond_r:  # Right lane
            if lanedir > 0:
                laneidx_goal_new = int(min(laneidx + 1, self.track.num_lane[segidx] - 1))
            else:
                laneidx_goal_new = int(max(laneidx - 1, 0))
        elif cond_l:  # Left lnae
            if lanedir > 0:
                laneidx_goal_new = int(max(laneidx - 1, 0))
            else:
                laneidx_goal_new = int(min(laneidx + 1, self.track.num_lane[segidx] - 1))
        else:  # Current lane
            laneidx_goal_new = int(laneidx)

        return laneidx_goal_new

    def get_goal_type(self, laneidx_goal, segidx, laneidx):
        """
        Gets goal type.
        :param laneidx_goal: lane-index of goal-point (int)
        :param segidx: current segment-index (int)
        :param laneidx: current lane-index (int)
        """
        lanedir = self.track.lane_dir[segidx][laneidx]
        diff_laneidx = laneidx_goal - laneidx
        if abs(diff_laneidx) == 0:
            goal_type_txt = 'C'
        else:
            if lanedir > 0:
                if diff_laneidx > 0:
                    goal_type_txt = 'R'
                else:
                    goal_type_txt = 'L'
            else:
                if diff_laneidx > 0:
                    goal_type_txt = 'L'
                else:
                    goal_type_txt = 'R'

        return goal_type_txt

    def get_goal_candidates(self, dim, pnt, segidx, laneidx, dist_goal_ahead):
        """
        Gets goal states candidates.
        :param dim: goal state dimension (2, 3, 4)
        :param pnt: current point (x, y) (dim = 2)
        :param segidx: segment index (int)
        :param laneidx: lane index (int)
        :param dist_goal_ahead: goal distance (float)
        """
        segidx, laneidx = int(segidx), int(laneidx)
        num_goal_candidates = 25
        states_goal = np.zeros((num_goal_candidates, dim), dtype=np.float32)
        pnt_c = get_lane_cp_wrt_mtrack(self.track, pnt, segidx, laneidx)

        for nidx_d in range(0, num_goal_candidates):
            dist_ahead_tmp = dist_goal_ahead - float(nidx_d) * (dist_goal_ahead / float(num_goal_candidates))
            pnt_goal = self.get_pnt_ahead(pnt_c[0:2], dist_ahead_tmp, segidx, laneidx)
            _, theta_goal = get_lane_cp_angle(pnt_goal[0:2], self.track.pnts_poly_track,
                                              self.track.pnts_lr_border_track)
            state_goal = np.zeros((dim,), dtype=np.float32)
            if dim == 2:
                state_goal[0:2] = pnt_goal
            elif dim == 3:
                state_goal[0:2], state_goal[2] = pnt_goal, theta_goal
            elif dim == 4:
                # state_goal[0:2], state_goal[2], state_goal[3] = pnt_goal, theta_goal, self.v_ref
                state_goal[0:2], state_goal[2], state_goal[3] = pnt_goal, theta_goal, 0.0
            else:
                pass

            states_goal[nidx_d, :] = state_goal

        return states_goal

    def check_goal_points(self, points_goal, data_ov):
        """
        Checks valid goal state.
        :param points_goal: goal states (dim = N x 2)
        :param data_ov: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
        """
        is_collision_array = np.zeros((points_goal.shape[0],), dtype=np.int32)
        if len(data_ov) > 0:
            for nidx_d in range(0, points_goal.shape[0]):
                state_goal_sel = points_goal[nidx_d, 0:2]
                state_goal_sel = np.reshape(state_goal_sel, (1, -1))
                is_collision = check_collision_pnts(state_goal_sel, data_ov)
                is_collision_array[nidx_d] = is_collision

        return is_collision_array

    def get_goal(self, data_tv, data_ov, dist_goal_ahead):
        """
        Gets goal state.
        :param data_tv: (target) vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = 10, width > length)
        :param data_ov: (target) vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
        """
        segidx_goal = data_tv[7]
        laneidx_goal = data_tv[8]

        states_goal = self.get_goal_candidates(4, data_tv[1:3], segidx_goal, laneidx_goal, dist_goal_ahead)
        is_collision_goal = self.check_goal_points(states_goal[:, 0:2], data_ov)

        idx_found_cg = np.where(is_collision_goal == 1)
        idx_found_cg = idx_found_cg[0]
        if len(idx_found_cg) > 0:
            idx_found_sel = min(idx_found_cg[-1] + 1, states_goal.shape[0] - 1)
            if abs(laneidx_goal - data_tv[8]) > 0:
                idx_found_sel = min(idx_found_sel, states_goal.shape[0] - 1 - 6)
            state_goal = states_goal[idx_found_sel, :]
        else:
            state_goal = states_goal[0, :]

        return state_goal, states_goal

    # NAIVE CONTROL ---------------------------------------------------------------------------------------------------#
    def find_control_naive(self, u_set, horizon, data_ov, t, pose_goal, return_seglane=False):
        """
        Finds (naive) control.
        :param u_set: set of controls (ndarray, dim = N x 2)
        :param horizon: trajectory horizon (int)
        :param data_ov: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param t: time (float)
        :param pose_goal: goal pose (dim = 3)
        :param return_seglane: whether to return next seg & lane indexes (boolean)
        """
        u_set = make_numpy_array(u_set, keep_1dim=False)
        data_ov = make_numpy_array(data_ov, keep_1dim=False)

        len_u_set = u_set.shape[0]
        # h_horizon_c = math.floor(horizon * 3.0 / 5.0)  # Horizon to compute
        h_horizon_c = math.floor(horizon * 0.4)  # Horizon to compute

        check_outside_array = np.zeros((len_u_set,), dtype=np.int32)
        check_collision_array = np.zeros((len_u_set, ), dtype=np.int32)
        check_lv_array = np.zeros((len_u_set, ), dtype=np.int32)
        dist_array = np.zeros((len_u_set,), dtype=np.float64)
        cnt_check = 0

        # Save other vehicle data w.r.t. time
        data_vehicle_array = []
        if len(data_ov) > 0:
            for nidx_h in range(1, horizon + 1):
                # Select data (w.r.t. time)
                t_sel = t + nidx_h
                idx_sel_1_ = np.where(data_ov[:, 0] == t_sel)
                idx_sel_1 = idx_sel_1_[0]
                data_vehicle_sel = data_ov[idx_sel_1, :]

                data_vehicle_array.append(data_vehicle_sel)

        # Compute cost
        traj_array = []
        seglane_array = np.zeros((len_u_set, 2), dtype=np.int32)
        for nidx_u in range(0, len_u_set):
            u_sel = u_set[nidx_u, :]

            traj_tmp = self.get_traj_naive(u_sel[0:2], horizon)

            h_outside, h_collision, h_lv_outside = 0, 0, 0
            for nidx_h in range(1, h_horizon_c):
                is_outside, is_collision, is_lv_outside = 0, 0, 0

                x_tmp = traj_tmp[nidx_h, :]

                # Get indexes of seg & lane
                seg_tmp_, lane_tmp_ = get_index_seglane(x_tmp[0:2], self.track.pnts_poly_track)
                seg_tmp, lane_tmp = seg_tmp_[0], lane_tmp_[0]

                if nidx_h == 1:
                    seglane_array[nidx_u, :] = [seg_tmp, lane_tmp]

                # Check inside-track =---------------------------------------------------------------------------------#
                if seg_tmp == -1 or lane_tmp == -1:
                    h_outside = nidx_h + 1
                    is_outside = 1

                # Check collision -------------------------------------------------------------------------------------#
                if len(data_vehicle_array) > 0:
                    data_vehicle_sel = data_vehicle_array[nidx_h - 1]

                    # data_t: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
                    data_t = [0, x_tmp[0], x_tmp[1], x_tmp[2], u_sel[0], (self.ry * 1.5), (self.rx * 1.5),
                              seg_tmp, lane_tmp, -1]
                    is_collision = check_collision(data_t, data_vehicle_sel)

                    if is_collision == 1:
                        h_collision = nidx_h + 1

                # Check linear-velocity -------------------------------------------------------------------------------#
                is_lv_outside = (x_tmp[3] < self.v_range[0]) or (x_tmp[3] > self.v_range[1])
                if is_lv_outside:
                    h_lv_outside = nidx_h + 1

                if is_outside == 1 or is_collision == 1 or is_lv_outside == 1:
                    break

            # Check reward --------------------------------------------------------------------------------------------#
            diff_ahead = pose_goal[0:2] - traj_tmp[-1, 0:2]
            dist_ahead = norm(diff_ahead)

            # Update
            traj_array.append(traj_tmp)
            cnt_check = cnt_check + 1
            check_outside_array[cnt_check - 1] = h_outside
            check_collision_array[cnt_check - 1] = h_collision
            check_lv_array[cnt_check - 1] = h_lv_outside

            dist_array[cnt_check - 1] = dist_ahead

        # Choose trajectory
        idx_invalid_1 = np.where(check_outside_array >= 1)
        idx_invalid_2 = np.where(check_collision_array >= 1)
        idx_invalid_3 = np.where(check_lv_array >= 1)

        idx_invalid_ = np.concatenate((idx_invalid_1[0], idx_invalid_2[0]), axis=0)
        idx_invalid_ = np.concatenate((idx_invalid_, idx_invalid_3[0]), axis=0)
        if len(idx_invalid_) > 0:
            idx_invalid = np.unique(idx_invalid_)
        else:
            idx_invalid = idx_invalid_

        cost_array = dist_array
        cost_array_c = np.copy(cost_array)
        cost_array_c[idx_invalid] = 10000

        idx_min = np.argmin(cost_array_c)
        traj_sel = traj_array[int(idx_min)]
        seglane_sel = seglane_array[int(idx_min), :]
        seglane_sel = np.reshape(seglane_sel, -1)

        if return_seglane:
            return traj_sel, traj_array, cost_array, idx_invalid, seglane_sel
        else:
            return traj_sel, traj_array, cost_array, idx_invalid

    # CONTROL BY REWARD -----------------------------------------------------------------------------------------------#
    def find_control_by_reward(self, u_set, horizon, data_ov, t):
        """
        Finds control w.r.t. reward.
        :param u_set: set of controls (ndarray, dim = N x 2)
        :param horizon: trajectory horizon (int)
        :param data_ov: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param t: time (float)
        """
        u_set = make_numpy_array(u_set, keep_1dim=False)
        data_ov = make_numpy_array(data_ov, keep_1dim=False)
        len_u_set = u_set.shape[0]

        traj_array = []
        r_array = np.zeros((len_u_set, ), dtype=np.float64)

        for nidx_u in range(0, len_u_set):
            u_sel = u_set[nidx_u, :]
            traj_tmp = self.get_traj_naive(u_sel[0:2], horizon)
            traj_array.append(traj_tmp)

            for nidx_h in range(0, horizon):
                nidx_t = t + nidx_h + 1

                # Get next state
                s_ev_next = traj_tmp[nidx_h + 1, :]

                seg_ev_next, lane_ev_next, data_ov_next, data_ev_next, f_next, id_near_next, data_ov_near_all_next, \
                data_ov_near_next, _ = self.get_info_t_reward(nidx_t, s_ev_next, s_ev_next[3], data_ov, use_intp=0)

                # EPI-STEP4: GET REWARD -------------------------------------------------------------------------------#
                # Compute reward component
                r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal, dist2goal = \
                    self.compute_reward_component(nidx_t, data_ev_next, s_ev_next[3], u_sel[0], data_ov_near_all_next,
                                                  f_next[0], f_next[4], f_next[8])
                # Get reward
                r_cur = self.compute_reward(r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed,
                                            r_angular, r_goal)

                r_array[nidx_u] = r_array[nidx_u] + r_cur

        idx_max = np.argmax(r_array)
        traj_sel = traj_array[idx_max]
        r_sel = r_array[idx_max]

        return traj_sel, traj_array, r_sel, r_array

    def get_info_t_reward(self, t, pose_t, v_t, data_ov, use_intp=0):
        """
        Gets current info for reward computation.
        :param t: current time (float)
        :param pose_t: current pose (x, y, theta) (ndarray, dim = 3)
        :param v_t: current linear velocity (float)
        :param data_ov: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param use_intp: whether to use interpolation (feature) (boolean)
        """
        # Set indexes of seg & lane
        seg_ev_t_, lane_ev_t_ = get_index_seglane(pose_t[0:2], self.track.pnts_poly_track)
        seg_ev_t, lane_ev_t = seg_ev_t_[0], lane_ev_t_[0]

        # Select current vehicle data (w.r.t time)
        if len(data_ov) == 0:
            data_ov_t = []
        else:
            idx_sel = np.where(data_ov[:, 0] == t)
            data_ov_t = data_ov[idx_sel[0], :]

        # Set data vehicle ego
        # structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
        data_ev_t = np.array([t, pose_t[0], pose_t[1], pose_t[2], v_t, self.ry, self.rx, seg_ev_t, lane_ev_t, -1],
                             dtype=np.float64)

        # Get feature
        f_t, id_near_t, _, _, _, dist_cf = get_feature(self.track, data_ev_t, data_ov_t, use_intp=use_intp)

        # Get near other vehicles
        data_ov_near_all_t, _ = select_data_ids(data_ov, id_near_t)

        if len(data_ov_near_all_t) > 0:
            idx_sel = np.where(data_ov_near_all_t[:, 0] == t)
            data_ov_near_t = data_ov_near_all_t[idx_sel[0], :]
        else:
            data_ov_near_t = []

        return seg_ev_t, lane_ev_t, data_ov_t, data_ev_t, f_t, id_near_t, data_ov_near_all_t, data_ov_near_t, dist_cf

    def compute_reward_component(self, t, data_ev_next, lv, w, data_ov_near, lane_dev_rad, lane_dev_dist_scaled, dist_cf):
        """
        Computes reward components.
        :param t: current-time (float)
        :param data_ev_next: next vehicle data [t x y theta v length width segment lane id] (dim = 10)
        :param lv: current linear velocity (float)
        :param w: current angular velocity (float)
        :param data_ov_near: nearby vehicle data [t x y theta v length width segment lane id] (dim = N x 10)
        :param lane_dev_rad: lane-dev angle (float)
        :param lane_dev_dist_scaled: lane-dev distance (float)
        :param dist_cf: cf distance (float)
        """
        # 0: Check outside
        if data_ev_next[7] == -1 or data_ev_next[8] == -1:
            r_outside = 1
        else:
            r_outside = 0

        # 1: Check collision
        r_collision = check_collision_t(data_ev_next, data_ov_near, t + 1)

        # 2 ~ 6
        r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular = \
            self.compute_reward_component_rest(lv, w, lane_dev_rad, lane_dev_dist_scaled, dist_cf)

        # 7: Reach goal points
        dist2goal_, reach_goal_ = get_dist2goal(data_ev_next[1:3], data_ev_next[7], data_ev_next[8],
                                                self.track.indexes_goal, self.track.pnts_goal)
        dist2goal = min(dist2goal_)
        r_goal = max(reach_goal_)

        return r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal, dist2goal

    def compute_reward_component_rest(self, lv, w, lane_dev_rad, lane_dev_dist_scaled, dist_cf):
        """
        Computes reset of reward components.
        :param lv: linear velocity (float)
        :param w: angular velocity (float)
        :param lane_dev_rad: lane-dev angle (float)
        :param lane_dev_dist_scaled: scaled lane-dev distance (float)
        :param dist_cf: cf distance (float)
        """
        r_w_lanedev = self.r_weight[2]

        # 2: Lane dev (rad)
        if abs(lane_dev_rad) > self.dev_rad_th * 2:
            r_dev_rad = -10 / r_w_lanedev
        elif abs(lane_dev_rad) > self.dev_rad_th:
            r_dev_rad = -0.33 / r_w_lanedev
        else:
            r_dev_rad = (self.dev_rad_th - abs(lane_dev_rad)) * (self.dev_rad_th - abs(lane_dev_rad))
            # max: (dev_rad_th)^2, min: 0

        # 3: Lane dev (dist)
        r_dev_dist = (0.5 - abs(lane_dev_dist_scaled)) * (0.5 - abs(lane_dev_dist_scaled))  # max: 0.25, min: 0

        # 4: Center front dist
        if dist_cf > self.dist_cf_th:  # max: (dist_cf_th)^2, min: 0
            r_cf_dist = self.dist_cf_th * self.dist_cf_th
        else:
            r_cf_dist = dist_cf * dist_cf

        # 5: Move forward
        if (lv < self.v_range[0]) or (lv > self.v_range[1]):
            r_speed = -100
        else:
            r_speed = self.v_ref * self.v_ref - (self.v_ref - lv) * (self.v_ref - lv)  # min: 0, max: (lev_ref)^2

        # 6: Turn left or right
        r_angular = self.w_max * self.w_max - (w * w)  # min: 0, max: (av_max)^2

        return r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular

    def compute_reward(self, r_outside, r_collision, r_dev_rad, r_dev_dist, r_cf_dist, r_speed, r_angular, r_goal):
        """
        Computes reward.
        :param r_outside: reward outside-track (float)
        :param r_collision: reward collision (float)
        :param r_dev_rad: reward lane-dev-angle (float)
        :param r_dev_dist: reward lane-dev-distance (float)
        :param r_cf_dist: reward lane-cf-distance (float)
        :param r_speed: reward speed (float)
        :param r_angular: reward angular_velocity (float)
        :param r_goal: reward goal (float)
        """
        r_out = self.r_weight[0] * r_outside + self.r_weight[1] * r_collision + self.r_weight[2] * r_dev_rad + \
                self.r_weight[3] * r_dev_dist + self.r_weight[4] * r_cf_dist + self.r_weight[5] * r_speed + \
                self.r_weight[6] * r_angular + self.r_weight[7] * r_goal
        r_out = min(max(r_out, -1), + 1)

        return r_out
