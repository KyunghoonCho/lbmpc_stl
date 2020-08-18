from __future__ import print_function

import math
import numpy as np

from src.get_rgb import *
from src.utils import *
from src.utils_sim import *
from src.utils_stl import *
from gurobipy import *  # Use gurobi for optimization


class MPCSTL(object):
    """
    MODEL PREDICTIVE CONTROL UNDER STL-CONSTRAINTS (MPCSTL)
        - Reference:
            [1] Kyunghoon Cho and Songhwai Oh,
            "Learning-Based Model Predictive Control under Signal Temporal Logic Specifications,"
            in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), May 2018.

        - STL rules:
            1: Lane-observation (down, right)
            2: Lane-observation (up, left)
            3: Collision (front)
            4: Collision (others)
            5: Speed-limit
            6: Slow-down

        - System Dynamic: 4-state unicycle dynamics
            State: [x, y, theta, v]
            Control: [a, w]
            dot(x) = v * cos(theta)
            dot(y) = v * sin(theta)
            dot(theta) = v * kappa_1 * w
            dot(v) = kappa_2 * a
    """
    def __init__(self, dt, rx, ry, kappa, h, h_relax, v_ref, v_range, u_lb, u_hb,
                 dist_goal_ahead, c_end_0, c_end_1, c_u_0, c_u_1, dth_c_end, rth_c_u, v_th,
                 dw_use_until, dsth_use_until, rth_use_until,
                 until_t_s, until_t_a, until_t_b, until_v_th, until_d_th, until_lanewidth, verbose):
        """
        Initialize MPC object.
        """
        self.dt = float(dt)  # Time step
        self.rx, self.ry = float(rx), float(ry)  # Ego-vehicle size
        self.kappa = kappa  # Dynamic parameters
        self.h = int(h)  # MPC-horizon
        self.h_relax = int(h_relax)  # Horizon of stl-constraint relaxation

        self.dim_x = 4  # Dimension of state
        self.dim_u = 2  # Dimension of control
        self.maxnum_oc = 6  # Maximum number of other-vehicles to consider

        # Linear velocity parameters
        self.v_ref = v_ref  # Reference (linear) velocity
        self.v_range = v_range  # Range of (linear) velocity

        # Lower & Upper bounds for control
        self.u_lb, self.u_ub = u_lb, u_hb

        # Cost parameters
        self.dist_goal_ahead = dist_goal_ahead  # Distance to the goal state
        self.c_end, self.c_u = c_end_0, c_u_0
        self.mode_cend, self.c_end_0, self.c_end_1 = 0, c_end_0, c_end_1
        self.mode_cu, self.c_u_0, self.c_u_1 = 0, c_u_0, c_u_1
        self.dth_c_end, self.rth_c_u = dth_c_end, rth_c_u

        # Center-front vehicle
        # The collision-avoidance constraint with 'center-front' vehicle is adjusted in some cases.
        self.rmin_c_cf_in = 0  # Robustness slackness of collision-avoidance (center-front) (input)

        # Speed-limit-logic parameter
        self.v_th = v_th  # Velocity threshold

        # Until-logic parameters
        self.dw_use_until, self.dsth_use_until, self.rth_use_until = dw_use_until, dsth_use_until, rth_use_until
        self.until_t_s, self.until_t_a, self.until_t_b = until_t_s, until_t_a, until_t_b
        self.until_v_th_ref, self.until_d_th_ref = until_v_th, until_d_th
        self.until_lanewidth = until_lanewidth
        self.until_v_th, self.until_d_th = until_v_th, until_d_th

        # Initial state (ego-vehicle)
        self.xinit = np.zeros((self.dim_x,), dtype=np.float32)
        self.xinit_conv = np.zeros((self.dim_x,), dtype=np.float32)

        # Goal state (ego-vehicle)
        self.xgoal = np.zeros((self.dim_x,), dtype=np.float32)
        self.xgoal_conv = np.zeros((self.dim_x,), dtype=np.float32)

        # OPTIMIZATION (GUROBI) ---------------------------------------------------------------------------------------#
        # Set optimization model
        self.opt_model = None  # Optimization model

        # Set optimization parameters
        self.list_h, self.list_dim_u, self.list_dim_x = range(0, self.h), range(0, self.dim_u), range(0, self.dim_x)
        self.list_h_relaxed = range(h_relax, self.h)

        self.params_u_lb = {(i, j): 0 for i in self.list_h for j in self.list_dim_u}  # Lower bound (control)
        for nidx_i in self.list_h:
            self.params_u_lb[nidx_i, 0], self.params_u_lb[nidx_i, 1] = u_lb[0], u_lb[1]

        self.params_u_ub = {(i, j): 0 for i in self.list_h for j in self.list_dim_u}  # Upper bound (control)
        for nidx_i in self.list_h:
            self.params_u_ub[nidx_i, 0], self.params_u_ub[nidx_i, 1] = u_hb[0], u_hb[1]

        self.params_x_lb = {(i, j): 0 for i in self.list_h for j in self.list_dim_x}  # Lower bound (state)
        for nidx_i in self.list_h:
            self.params_x_lb[nidx_i, 0], self.params_x_lb[nidx_i, 1] = -1000.0, -1000.0
            self.params_x_lb[nidx_i, 2] = - math.pi / 2.0
            self.params_x_lb[nidx_i, 3] = self.v_range[0]

        self.params_x_ub = {(i, j): 0 for i in self.list_h for j in self.list_dim_x}  # Upper bound (state)
        for nidx_i in self.list_h:
            self.params_x_ub[nidx_i, 0], self.params_x_ub[nidx_i, 1] = +1000.0, +1000.0
            self.params_x_ub[nidx_i, 2] = + math.pi / 2.0
            self.params_x_ub[nidx_i, 3] = self.v_range[1]

        self.cp_l_d, self.cp_l_u = [], []  # Lane-constraints (point)
        self.rad_l = 0.0  # Lane-constraints (angle)

        self.id_ov_near = -1.0 * np.ones((6,))  # Id of near-vehicles [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        self.traj_ov_cf = np.zeros((1, 3))  # Collision-constraints (trajectory): id_cf
        self.size_ov_cf = np.zeros((1, 2))  # Collision-constraints (size: dx, dy): id_cf
        self.traj_ov = []  # Collision-constraints (trajectory): list [id_lf, id_lr, id_rf, id_rr, id_cr]
        self.size_ov = []  # Collision-constraints (size: dx, dy): list [id_lf, id_lr, id_rf, id_rr, id_cr]

        # Initial computed robustness slackness
        self.r_l_down_init, self.r_l_up_init, self.r_c_cf_init = 0.0, 0.0, 0.0
        self.r_c_rest_array_init = []

        # Set variables
        self.u_vars, self.x_vars = [], []
        self.z_c_cf_vars, self.r_c_cf_vars = [], []
        self.z_c_ov_vars, self.r_c_ov_vars = [], []

        self.r_until, self.r_until_and1 = 0, 0
        self.z_until_and1, self.z_until_and2 = [], []
        self.r_until_and_hist, self.z_until_and_hist = [], []
        self.r_until_or_hist, self.z_until_or_hist = [], []
        self.r_until_alw, self.z_until_alw = 0, []
        self.r_until_ev, self.z_until_ev = 0, []

        self.M = float(1e5)  # Big value (mixed-integer programming)
        self.maxnum_trial = 3  # Maximum number of tiral

        # FOR DEBUG ---------------------------------------------------------------------------------------------------#
        self.verbose = verbose
        self.cp_l_d_rec, self.cp_l_u_rec = [], []  # Lane-constraints (point)
        self.traj_l_d_rec, self.traj_l_u_rec = [], []

        # History
        self.cnt_hist_ev = 0
        self.x_hist_ev = np.zeros((10000, 4), dtype=np.float32)
        self.xgoal_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpd_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpu_hist_ev = np.zeros((10000, 2), dtype=np.float32)

        self.x_conv_hist_ev = np.zeros((10000, 4), dtype=np.float32)
        self.xgoal_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpd_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)
        self.cpu_conv_hist_ev = np.zeros((10000, 2), dtype=np.float32)

        # FOR PLOT ----------------------------------------------------------------------------------------------------#
        self.hcolor_lane_d = get_rgb("Orange")
        self.hcolor_lane_u = get_rgb("Green Yellow")

    # RESET MODEL -----------------------------------------------------------------------------------------------------#
    def reset_opt_model(self):
        """
        Resets optimization model
        """
        self.opt_model = Model("MPCSTL")
        self.opt_model.setParam('OutputFlag', False)  # Whether to be verbose

        # # (Primal feasibility tolerance) Tightening this tolerance can produce smaller constraint violations,
        # # but for numerically challenging models it can sometimes lead to much larger iteration counts.
        # self.opt_model.setParam('FeasibilityTol', float(1e-2))  # Default: 1e-6, min: 1e-9, max: 1e-2
        #
        # # (Dual feasibility tolerance) Reduced costs must all be smaller than OptimalityTol
        # # in the improving direction in order for a model to be declared optimal.
        # self.opt_model.setParam('OptimalityTol', float(1e-3))  # Default: 1e-6, min: 1e-9, max: 1e-2
        #
        # # (Integer feasibility tolerance) Tightening this tolerance can produce smaller integrality violations,
        # # but very tight tolerances may significantly increase runtime. Loosening this tolerance rarely reduces runtime.
        # self.opt_model.setParam('IntFeasTol', float(1e-1))  # Default: 1e-5, min: 1e-9, max: 1e-1
        #
        # # (Relative MIP optimality gap) The MIP solver will terminate (with an optimal result) when the gap
        # # between the lower and upper objective bound is less than MIPGap times the absolute value of the upper bound.
        # self.opt_model.setParam('MIPGap', 0.01)  # Default: 1e-4, min: 0, max: Infty

        # # self.opt_model.setParam('MIPFocus', 1)
        return

    # CONVERT STATE ---------------------------------------------------------------------------------------------------#
    def convert_state(self, x, xgoal, cp2rotate, theta2rotate):
        """
        Rotates w.r.t. reference pose.
        :param x: ego-vehicle system state (dim = 4)
        :param xgoal: goal system state (dim = 4)
        :param cp2rotate: center point to convert (dim = 2)
        :param theta2rotate: angle (rad) to convert (float)
        """
        # Convert state (ego-vehicle)
        self.xinit = x
        self.xinit_conv = convert_state(x, cp2rotate, theta2rotate)

        # Convert state (goal)
        if len(xgoal) > 0:
            self.xgoal = xgoal
            self.xgoal_conv = convert_state(xgoal, cp2rotate, theta2rotate)
        else:
            xgoal_conv_in = np.zeros((self.dim_x,), dtype=np.float32)
            xgoal_conv_in[0] = self.dist_goal_ahead
            self.xgoal_conv = xgoal_conv_in

            self.xgoal = np.zeros((self.dim_x,), dtype=np.float32)
            pnt_goal_tmp = get_rotated_pnts_rt(xgoal_conv_in[0:2], +cp2rotate, +theta2rotate)
            self.xgoal[0:2] = pnt_goal_tmp

    # SET GOAL --------------------------------------------------------------------------------------------------------#
    def get_goal_laneidx(self, rpred_l_r, rpred_l_l, r_l_th, segidx, laneidx, sim_control):
        """
        Gets goal lane-index.
        :param rpred_l_r: robustness slackness prediction (lane-observation-right) (float)
        :param rpred_l_l: robustness slackness prediction (lane-observation-left) (float)
        :param r_l_th: robustness slackness threshold (float)
        :param segidx: current segment-index (int)
        :param laneidx: current lane-index (int)
        :param sim_control: control-info
        """
        cond_r, cond_l = False, False
        if rpred_l_r < -r_l_th:  # Right lane
            cond_r = True
        elif rpred_l_l < -r_l_th:  # Left lnae
            cond_l = True
        else:  # Current lane
            pass

        laneidx_goal_new = sim_control.get_goal_laneidx(cond_r, cond_l, segidx, laneidx)

        return laneidx_goal_new

    def get_goal_state(self, toggle_goal, rpred_l_r, rpred_l_l, r_l_th, laneidx_goal, s_ev_cur, segidx, laneidx,
                       data_ov_cur, traj_ov_cf, size_ov_cf, sim_control, trackname):
        """
        Gets goal state.
        :param toggle_goal: whether to update goal-state (boolean)
        :param rpred_l_r: robustness slackness prediction (lane-observation-right) (float)
        :param rpred_l_l: robustness slackness prediction (lane-observation-left) (float)
        :param r_l_th: robustness slackness threshold (float)
        :param laneidx_goal: lane-index of goal-point (int)
        :param s_ev_cur: current ego-vehicle state
        :param segidx: current segment-index (int)
        :param laneidx: current lane-index (int)
        :param data_ov_cur: current other-vehicles
        :param traj_ov_cf: trajectory of (cf) other-vehicle.
        :param size_ov_cf: size of (cf) other-vehicle.
        :param sim_control: control-info
        :param trackname: track-name
        """
        if toggle_goal:
            laneidx_goal_new = self.get_goal_laneidx(rpred_l_r, rpred_l_l, r_l_th, segidx, laneidx, sim_control)

            if "us101" in trackname and segidx == 0 and laneidx == 4 and laneidx_goal_new == 5:
                toggle_goal = False
            elif "us101" in trackname and segidx == 2 and laneidx == 4 and laneidx_goal_new == 5:
                toggle_goal = False
            elif abs(laneidx_goal_new - laneidx_goal) > 0:
                laneidx_goal = laneidx_goal_new
                toggle_goal = False

        goal_type_txt = sim_control.get_goal_type(laneidx_goal, segidx, laneidx)

        states_goal = sim_control.get_goal_candidates(4, s_ev_cur[0:2], segidx, laneidx_goal, self.dist_goal_ahead)
        is_collision_goal = sim_control.check_goal_points(states_goal[:, 0:2], data_ov_cur)

        # data_ov: vehicle data [t x y theta v length width tag_segment tag_lane id] (dim = N x 10, width > length)
        # data_ov_cf = np.zeros((1, 10), dtype=np.float32)
        # data_ov_cf[0, 1:4] = traj_ov_cf[5, 0:3]
        # data_ov_cf[0, 5:7] = [size_ov_cf[1], size_ov_cf[0]]
        # is_collision_goal = sim_control.check_goal_points(states_goal[:, 0:2], data_ov_cf)

        idx_found_cg = np.where(is_collision_goal == 1)
        idx_found_cg = idx_found_cg[0]
        if len(idx_found_cg) > 0:
            idx_found_sel = min(idx_found_cg[-1] + 1, states_goal.shape[0] - 1)
            if abs(laneidx_goal - laneidx) > 0:
                idx_found_sel = min(idx_found_sel, states_goal.shape[0] - 1 - 6)
            state_goal = states_goal[idx_found_sel, :]
        else:
            state_goal = states_goal[0, :]

        return toggle_goal, laneidx_goal, goal_type_txt, states_goal, state_goal

    # GET CONSTRAINTS -------------------------------------------------------------------------------------------------#
    def get_lane_constraints(self, pnt_down, pnt_up, lane_angle, margin_dist, cp2rotate, theta2rotate):
        """
        Gets lane constraints.
        :param pnt_down: point (ndarray, dim = 2)
        :param pnt_up: point (ndarray, dim = 2)
        :param lane_angle: lane-heading (rad) (float)
        :param margin_dist: margin dist (bigger -> loosing constraints) (float)
        :param cp2rotate: center point to convert (ndarray, dim = 2)
        :param theta2rotate: angle (rad) to convert (float)
        """
        self.cp_l_d, self.cp_l_u, self.rad_l = [], [], 0.0

        pnt_down_r = np.reshape(pnt_down[0:2], (1, 2))
        pnt_up_r = np.reshape(pnt_up[0:2], (1, 2))
        pnt_down_conv_ = get_rotated_pnts_tr(pnt_down_r, -cp2rotate, -theta2rotate)
        pnt_up_conv_ = get_rotated_pnts_tr(pnt_up_r, -cp2rotate, -theta2rotate)
        pnt_down_conv, pnt_up_conv = pnt_down_conv_[0, :], pnt_up_conv_[0, :]
        lane_angle_r = angle_handle(lane_angle - theta2rotate)

        margin_dist = margin_dist - self.ry / 2.0
        self.cp_l_d = pnt_down_conv + np.array([+margin_dist*math.sin(lane_angle_r),
                                                -margin_dist*math.cos(lane_angle_r)], dtype=np.float32)
        self.cp_l_u = pnt_up_conv + np.array([-margin_dist * math.sin(lane_angle_r),
                                              +margin_dist * math.cos(lane_angle_r)], dtype=np.float32)
        self.rad_l = lane_angle_r

        self.cp_l_d_rec = get_rotated_pnts_rt(self.cp_l_d, +cp2rotate, +theta2rotate)
        self.cp_l_u_rec = get_rotated_pnts_rt(self.cp_l_u, +cp2rotate, +theta2rotate)

        xtmp = np.arange(start=-30, stop=+30, step=0.5)
        len_xtmp = xtmp.shape[0]
        y_lower = self.cp_l_d[1] * np.ones((len_xtmp,), dtype=np.float32)
        y_upper = self.cp_l_u[1] * np.ones((len_xtmp,), dtype=np.float32)
        traj_lower = np.zeros((len_xtmp, 2), dtype=np.float32)
        traj_upper = np.zeros((len_xtmp, 2), dtype=np.float32)
        traj_lower[:, 0] = xtmp
        traj_lower[:, 1] = y_lower
        traj_upper[:, 0] = xtmp
        traj_upper[:, 1] = y_upper

        self.traj_l_d_rec = get_rotated_pnts_rt(traj_lower, +cp2rotate, +theta2rotate)
        self.traj_l_u_rec = get_rotated_pnts_rt(traj_upper, +cp2rotate, +theta2rotate)

    def get_collision_constraints(self, traj_ov_near_list, size_ov_near_list, id_near, cp2rotate, theta2rotate, lw):
        """
        Gets collision constraints.
        :param traj_ov_near_list: (list)-> x y theta (ndarray, dim = N x 3)
                                  [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        :param size_ov_near_list: (list)-> (ndarray) dx dy (dim = N x 2)
                                  [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr]
        :param id_near: near ids [id_lf, id_lr, id_rf, id_rr, id_cf, id_cr] (ndarray, dim = 6)
        :param cp2rotate: center point to convert (ndarray, dim = 2)
        :param theta2rotate: angle (rad) to convert (float)
        :param lw: lanewidth (float)
        """
        # Do reset
        self.traj_ov, self.size_ov = [], []
        self.id_ov_near = id_near

        len_ov_near = len(traj_ov_near_list)

        for nidx_l in range(0, len_ov_near):
            traj_ov_near_list_sel = traj_ov_near_list[nidx_l]
            size_ov_near_list_sel = size_ov_near_list[nidx_l]
            id_near_sel = id_near[nidx_l]

            if nidx_l == 4:  # id_cf
                if id_near_sel == -1:
                    self.traj_ov_cf = traj_ov_near_list_sel
                    self.size_ov_cf = np.zeros((self.h + 1, 2), dtype=np.float32)
                else:
                    traj_tmp = np.zeros((self.h + 1, 3), dtype=np.float32)
                    traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                    traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                    diff_tmp = traj_tmp[:, 0:2] - np.reshape(self.xinit_conv[0:2], (1, 2))
                    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                    idx_tmp_ = np.where(dist_tmp > 100.0)
                    idx_tmp_ = idx_tmp_[0]
                    if len(idx_tmp_) > 0:
                        traj_tmp[idx_tmp_, 0:2] = [self.xinit_conv[0] - 100, self.xinit_conv[1] - 100]

                    self.traj_ov_cf = traj_tmp

                    self.size_ov_cf = np.zeros((self.h + 1, 2), dtype=np.float32)
                    for nidx_h in range(0, self.h + 1):
                        traj_tmp_sel = traj_tmp[nidx_h, :]
                        size_ov_near_list_sel_new = np.array([size_ov_near_list_sel[0], size_ov_near_list_sel[1]],
                                                             dtype=np.float32)

                        # if size_ov_near_list_sel_new[1] < lanewidth * 0.9:
                        #     size_ov_near_list_sel_new[1] = lanewidth * 0.9

                        size_tmp_sel = self.get_modified_size_linear(size_ov_near_list_sel_new, traj_tmp_sel[2])
                        self.size_ov_cf[nidx_h, :] = size_tmp_sel

            else:
                if id_near_sel == -1:
                    traj_tmp = traj_ov_near_list_sel
                    size_tmp = np.zeros((self.h + 1, 2), dtype=np.float32)
                else:
                    traj_tmp = np.zeros((self.h + 1, 3), dtype=np.float32)
                    traj_tmp[:, 0:2] = get_rotated_pnts_tr(traj_ov_near_list_sel[:, 0:2], -cp2rotate, -theta2rotate)
                    traj_tmp[:, 2] = angle_handle(traj_ov_near_list_sel[:, 2] - theta2rotate)

                    diff_tmp = traj_tmp[:, 0:2] - np.reshape(self.xinit_conv[0:2], (1, 2))
                    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
                    idx_tmp_ = np.where(dist_tmp > 100.0)
                    idx_tmp_ = idx_tmp_[0]
                    if len(idx_tmp_) > 0:
                        traj_tmp[idx_tmp_, 0:2] = [self.xinit_conv[0] - 100, self.xinit_conv[1] - 100]

                    size_tmp = np.zeros((self.h + 1, 2), dtype=np.float32)
                    for nidx_h in range(0, self.h + 1):
                        traj_tmp_sel = traj_tmp[nidx_h, :]
                        size_tmp_sel = self.get_modified_size_linear(size_ov_near_list_sel, traj_tmp_sel[2])
                        size_tmp[nidx_h, :] = size_tmp_sel

                self.traj_ov.append(traj_tmp)
                self.size_ov.append(size_tmp)

    # MODEL PREDICTIVE CONTROL ----------------------------------------------------------------------------------------#
    # SOLVE BY MIXED INTEGER PROGRAMMING ------------------------------------------------------------------------------#
    def control_by_mpc(self, rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until, cp2rotate,
                       theta2rotate, use_until, lw):
        """
        Computes control by MPC.
        :param rmin_l_d: robustness slackness lane-down (float)
        :param rmin_l_u: robustness slackness lane-up (float)
        :param rmin_c_cf: robustness slackness collision-centerfront (float)
        :param rmin_c_rest: robustness slackness collision-rest (float)
        :param rmin_speed: robustness slackness collision-speed (float)
        :param rmin_until: robustness slackness until (float)
        :param cp2rotate: center point to convert (ndarray, dim = 2)
        :param theta2rotate: angle (rad) to convert (float)
        :param use_until: use until-logic (boolean)
        :param lw: lanewidth (float)
        """
        if use_until:
            self.until_v_th = self.until_v_th_ref / self.until_lanewidth * lw
            self.until_d_th = self.until_d_th_ref / self.until_lanewidth * lw

        is_error, is_error_until = 0, 0
        rmin_l_mod, rmin_s_mod = 0, 0
        for nidx_trial in range(0, self.maxnum_trial):
            if self.verbose == 2:
                print("TRIAL: {:d}".format(nidx_trial + 1))
            is_error, x_out, u_out, _is_error_until = \
                self.solve_by_mip(rmin_l_d + rmin_l_mod, rmin_l_u + rmin_l_mod, rmin_c_cf, rmin_c_rest,
                                  rmin_speed + rmin_s_mod, rmin_until, use_until=use_until)

            if is_error == 0:
                if use_until and self.verbose == 2:
                    self.print_r_until()
                break
            else:
                if _is_error_until == 1:
                    is_error_until = 1 if is_error_until == 0 else 0
                    use_until = 0
                else:
                    rmin_l_mod = rmin_l_mod - 1 * 0.08 * lw
                    rmin_s_mod = rmin_s_mod - self.v_th / 20

        if is_error == 1:
            if self.verbose == 2:
                print("FAILED TO SOLVE")
            xinit_conv_r = np.reshape(self.xinit_conv, (1, -1))
            x_out = np.tile(xinit_conv_r, [self.h + 1, 1])
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)

        x_out_revert = np.zeros((self.h + 1, self.dim_x), dtype=np.float32)
        x_out_revert[:, 0:2] = get_rotated_pnts_rt(x_out[:, 0:2], +cp2rotate, +theta2rotate)
        x_out_revert[:, 2] = angle_handle(x_out[:, 2] + theta2rotate)
        x_out_revert[:, 3] = x_out[:, 3]

        return x_out, u_out, x_out_revert, (nidx_trial + 1), is_error, is_error_until

    def solve_by_mip(self, rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed, rmin_until, use_until=1):
        """
        Solves MPC by mixed integer programming.
        :param rmin_l_d: robustness slackness lane-down (float)
        :param rmin_l_u: robustness slackness lane-up (float)
        :param rmin_c_cf: robustness slackness collision-centerfront (float)
        :param rmin_c_rest: robustness slackness collision-rest (float)
        :param rmin_speed: robustness slackness collision-speed (float)
        :param rmin_until: robustness slackness until (float)
        :param use_until: whether to use until constraint (boolean)
        """
        self.reset_opt_model()  # Reset model

        # GET ROBUSTNESS SLACKNESS ------------------------------------------------------------------------------------#
        rmin_l_d_c, rmin_l_u_c, rmin_c_cf_c, rmin_c_rest_c, rmin_speed_c = \
            self.compute_robustness_slackness_init(rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_speed)

        # SET VARIABLES -----------------------------------------------------------------------------------------------#
        u_dict = {(i, j) for i in self.list_h for j in self.list_dim_u}
        self.u_vars = self.opt_model.addVars(u_dict, lb=self.params_u_lb, ub=self.params_u_ub, vtype=GRB.CONTINUOUS, name="u")
        # self.u_vars = self.opt_model.addVars(u_dict, vtype=GRB.CONTINUOUS, name="u")

        x_dict = {(i, j) for i in self.list_h for j in self.list_dim_x}
        self.x_vars = self.opt_model.addVars(x_dict, lb=self.params_x_lb, ub=self.params_x_ub, vtype=GRB.CONTINUOUS, name="x")
        # self.x_vars = self.opt_model.addVars(x_dict, vtype=GRB.CONTINUOUS, name="x")

        z_c_dict = {(i, j) for i in self.list_h for j in range(0, 4)}
        r_c_dict = {i for i in self.list_h}
        self.z_c_cf_vars = self.opt_model.addVars(z_c_dict, vtype=GRB.BINARY, name="z_c_cf")
        self.r_c_cf_vars = self.opt_model.addVars(r_c_dict, vtype=GRB.CONTINUOUS, name="r_c_cf")

        self.z_c_ov_vars, self.r_c_ov_vars = [], []
        for nidx_oc in range(0, self.maxnum_oc - 1):
            txt_labelz, txt_labelr = "z_c_" + str(nidx_oc), "r_c_" + str(nidx_oc)
            z_c_ov_vars_ = self.opt_model.addVars(z_c_dict, vtype=GRB.BINARY, name=txt_labelz)
            r_c_ov_vars_ = self.opt_model.addVars(r_c_dict, vtype=GRB.CONTINUOUS, name=txt_labelr)
            self.z_c_ov_vars.append(z_c_ov_vars_)
            self.r_c_ov_vars.append(r_c_ov_vars_)

        if use_until == 1:
            min_until_init, max_until_init = -500.0, +500.0
            self.r_until = self.opt_model.addVar(lb=min_until_init, ub=max_until_init, vtype=GRB.CONTINUOUS, name="r_until")
            self.r_until_and1 = self.opt_model.addVar(lb=min_until_init, ub=max_until_init, vtype=GRB.CONTINUOUS, name="r_until_and1")
            until_dict0_z = {i for i in range(0, 2)}
            self.z_until_and1 = self.opt_model.addVars(until_dict0_z, vtype=GRB.BINARY, name="z_until_and1")
            self.z_until_and2 = self.opt_model.addVars(until_dict0_z, vtype=GRB.BINARY, name="z_until_and2")

            until_dict1_z = {(i, j) for i in range(0, self.until_t_b - self.until_t_a) for j in range(0, 2)}
            until_dict1_r = {i for i in range(0, self.until_t_b - self.until_t_a)}
            self.z_until_and_hist = self.opt_model.addVars(until_dict1_z, vtype=GRB.BINARY, name="z_until_and_hist")
            self.r_until_and_hist = self.opt_model.addVars(until_dict1_r, lb=min_until_init, ub=max_until_init, vtype=GRB.CONTINUOUS, name="r_until_and_hist")
            until_dict2_z = {(i, j) for i in range(0, self.until_t_b - self.until_t_a) for j in range(0, 2)}
            until_dict2_r = {i for i in range(0, self.until_t_b - self.until_t_a)}
            self.z_until_or_hist = self.opt_model.addVars(until_dict2_z, vtype=GRB.BINARY, name="z_until_or_hist")
            self.r_until_or_hist = self.opt_model.addVars(until_dict2_r, lb=min_until_init, ub=max_until_init, vtype=GRB.CONTINUOUS,
                                                          name="r_until_or_hist")

            alw_dic_z = {i for i in range(0, self.until_t_a - self.until_t_s)}
            self.z_until_alw = self.opt_model.addVars(alw_dic_z, vtype=GRB.BINARY, name="z_until_alw")
            self.r_until_alw = self.opt_model.addVar(vtype=GRB.CONTINUOUS, name="r_until_alw")

            ev_dic_z = {i for i in range(0, self.until_t_b + 1 - self.until_t_a)}
            self.z_until_ev = self.opt_model.addVars(ev_dic_z, vtype=GRB.BINARY, name="z_until_ev")
            self.r_until_ev = self.opt_model.addVar(lb=min_until_init, ub=max_until_init, vtype=GRB.CONTINUOUS, name="r_until_ev")

        self.opt_model.update()  # Update model

        # SET OBJECTIVE -----------------------------------------------------------------------------------------------#
        cost_opt = 0.0
        for nidx_d in range(0, self.dim_x):
            cost_opt += (self.x_vars[(self.h - 1, nidx_d)] - float(self.xgoal_conv[nidx_d])) * \
                        (self.x_vars[(self.h - 1, nidx_d)] - float(self.xgoal_conv[nidx_d])) * float(self.c_end[nidx_d])

        for idx_h in range(0, self.h):
            for nidx_d in range(0, self.dim_u):
                cost_opt += (self.u_vars[(idx_h, nidx_d)]) * (self.u_vars[(idx_h, nidx_d)]) * float(self.c_u[nidx_d])

        self.opt_model.setObjective(cost_opt, GRB.MINIMIZE)

        # SET CONSTRAINTS ---------------------------------------------------------------------------------------------#
        # Set constraints (dynamic)
        self.set_dynamic_constraints()

        # Set constraints (lane)
        self.set_lane_constraints(rmin_l_d_c, rmin_l_u_c)

        # Set constraints (collision)
        id_ov_near_cf = self.id_ov_near[4]
        id_ov_near_rest = self.id_ov_near[[0, 1, 2, 3, 5]]

        if id_ov_near_cf >= 0:
            self.set_collision_constraints(self.x_vars, self.z_c_cf_vars, self.r_c_cf_vars, rmin_c_cf_c,
                                           self.traj_ov_cf, self.size_ov_cf, "c_cf")

        for nidx_oc in range(0, self.maxnum_oc - 1):  # [id_lf, id_lr, id_rf, id_rr, id_cr]
            if id_ov_near_rest[nidx_oc] >= 0:
                txt_label = "c_ov" + str(nidx_oc)
                self.set_collision_constraints(self.x_vars, self.z_c_ov_vars[nidx_oc], self.r_c_ov_vars[nidx_oc],
                                               rmin_c_rest_c[nidx_oc], self.traj_ov[nidx_oc], self.size_ov[nidx_oc],
                                               txt_label)

        # Set constraints (speed)
        self.set_speed_constraints(rmin_speed_c)

        # Set constraints (until)
        if use_until == 1:
            self.set_until_constraint(rmin_until)

        self.opt_model.update()  # Update model

        # OPTIMIZE MODEL ----------------------------------------------------------------------------------------------#
        self.opt_model.optimize()
        is_error = self.check_opt_status()

        is_error_until = 0
        if is_error == 1:
            self.opt_model.computeIIS()  # Computes Irreducible Incosistent Subsytems (Not Vital)

            if self.verbose == 2:
                print("\nThe following constraint(s) cannot be satisfied:")
            for c in self.opt_model.getConstrs():
                if c.IISConstr:
                    if self.verbose == 2:
                        print('%s' % c.constrName)
                    if "until" in c.constrName:
                        is_error_until = 1

        # Set output
        if is_error == 1:
            xinit_conv_r = np.reshape(self.xinit_conv, (1, -1))
            x_out = np.tile(xinit_conv_r, [self.h + 1, 1])
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)
        else:
            x_out = np.zeros((self.h + 1, self.dim_x), dtype=np.float32)
            u_out = np.zeros((self.h, self.dim_u), dtype=np.float32)
            x_out[0, :] = self.xinit_conv
            for idx_h in range(0, self.h):
                x_out[idx_h + 1, :] = [self.x_vars[(idx_h, 0)].X, self.x_vars[(idx_h, 1)].X,
                                        self.x_vars[(idx_h, 2)].X, self.x_vars[(idx_h, 3)].X]
                u_out[idx_h, :] = [self.u_vars[(idx_h, 0)].X, self.u_vars[(idx_h, 1)].X]

            if use_until == 1 and self.verbose == 2:
                print("r_until: {:.3f}".format(self.r_until.X))

        return is_error, x_out, u_out, is_error_until

    def check_opt_status(self):
        """
        Checks optimization status.
        """
        is_error = 0
        if self.opt_model.status == GRB.Status.OPTIMAL:
            if self.verbose >= 2:
                print('Optimal objective: %g' % self.opt_model.objVal)
        elif self.opt_model.status == GRB.Status.INF_OR_UNBD:
            if self.verbose >= 1:
                print('Model is infeasible or unbounded')
            is_error = 1
        elif self.opt_model.status == GRB.Status.INFEASIBLE:
            if self.verbose >= 1:
                print('Model is infeasible')
            is_error = 1
        elif self.opt_model.status == GRB.Status.UNBOUNDED:
            if self.verbose >= 1:
                print('Model is unbounded')
            is_error = 1
        else:
            if self.verbose >= 1:
                print('Optimization ended with status %d' % self.opt_model.status)
        return is_error

    def set_control_cost_mode(self, dist_cf, lanedev_rad):
        """
        Sets control-cost mode (modify cost).
        :param dist_cf: distance to center-front vehicle (float)
        :param lanedev_rad: lane-deviation angle (float)
        """
        if abs(dist_cf) < self.dth_c_end:
            self.c_end = self.c_end_1
            self.mode_cend = 1
        else:
            self.c_end = self.c_end_0
            self.mode_cend = 0

        if abs(lanedev_rad) > self.rth_c_u:
            self.c_u = self.c_u_1
            self.mode_cu = 1
        else:
            self.c_u = self.c_u_0
            self.mode_cu = 0

    def set_use_until(self, id_near, dist_cf, lanewidth, lanedev_dist_scaled, lanedev_rad, laneidx, laneidx_goal):
        """
        Sets whether to apply until-logic.
        """
        __until_d_th = self.until_d_th / self.until_lanewidth * lanewidth * self.dw_use_until
        __cond1 = (laneidx_goal != laneidx) and ((abs(lanedev_dist_scaled) > self.dsth_use_until) or
                                                 (abs(lanedev_rad) > self.rth_use_until))
        __cond2 = (dist_cf < __until_d_th) and (id_near[4] >= 0)

        if __cond1:
            # Until-logic is not considered when changing lanes.
            use_until = 0
        elif __cond2:
            # Until-logic is difficult to handle, so it is handled within a certain distance from the vehicle in front.
            use_until = 1
        else:
            use_until = 0

        return use_until

    # SET CONSTRAINTS -------------------------------------------------------------------------------------------------#
    def set_dynamic_constraints(self):
        """
        Sets dynamic constraints.
        """
        dt = self.dt
        kappa1, kappa2 = self.kappa[0], self.kappa[1]
        theta_ref, v_ref = float(self.xinit_conv[2]), float(self.xinit_conv[3])
        cos_ref, sin_ref = math.cos(theta_ref), math.sin(theta_ref)

        xinit_conv_x, xinit_conv_y, xinit_conv_theta, xinit_conv_v = \
            float(self.xinit_conv[0]), float(self.xinit_conv[1]), float(self.xinit_conv[2]), float(self.xinit_conv[3])

        a02, a03 = -1 * v_ref * sin_ref * dt, cos_ref * dt
        a12, a13 = v_ref * cos_ref * dt, sin_ref * dt

        b0 = v_ref * kappa1 * dt
        b1 = kappa2 * dt
        c0 = v_ref * sin_ref * dt * theta_ref
        c1 = -v_ref * cos_ref * dt * theta_ref

        rhs00 = xinit_conv_x + a02 * xinit_conv_theta + a03 * xinit_conv_v + c0
        rhs01 = xinit_conv_y + a12 * xinit_conv_theta + a13 * xinit_conv_v + c1

        for idx_h in range(0, self.h):
            if idx_h == 0:
                self.opt_model.addConstr(self.x_vars[(idx_h, 0)] == rhs00, "dyn0_{:d}".format(idx_h))
                self.opt_model.addConstr(self.x_vars[(idx_h, 1)] == rhs01, "dyn1_{:d}".format(idx_h))
                self.opt_model.addConstr(self.x_vars[(idx_h, 2)] == xinit_conv_theta + b0 * self.u_vars[(idx_h, 0)],
                                         "dyn2_{:d}".format(idx_h))
                self.opt_model.addConstr(self.x_vars[(idx_h, 3)] == xinit_conv_v + b1 * self.u_vars[(idx_h, 1)],
                                         "dyn3_{:d}".format(idx_h))
            else:
                txt_dyn_tmp1 = "dyn0_{:d}".format(idx_h)
                self.opt_model.addConstr(self.x_vars[(idx_h, 0)] == self.x_vars[(idx_h - 1, 0)] +
                                         a02 * self.x_vars[(idx_h - 1, 2)] + a03 * self.x_vars[(idx_h - 1, 3)] +
                                         c0, txt_dyn_tmp1)
                txt_dyn_tmp2 = "dyn1_{:d}".format(idx_h)
                self.opt_model.addConstr(self.x_vars[(idx_h, 1)] == self.x_vars[(idx_h - 1, 1)] +
                                         a12 * self.x_vars[(idx_h - 1, 2)] + a13 * self.x_vars[(idx_h - 1, 3)] + c1,
                                         txt_dyn_tmp2)
                txt_dyn_tmp3 = "dyn2_{:d}".format(idx_h)
                self.opt_model.addConstr(self.x_vars[(idx_h, 2)] == self.x_vars[(idx_h - 1, 2)] +
                                         b0 * self.u_vars[(idx_h, 0)], txt_dyn_tmp3)
                txt_dyn_tmp4 = "dyn3_{:d}".format(idx_h)
                self.opt_model.addConstr(self.x_vars[(idx_h, 3)] == self.x_vars[(idx_h - 1, 3)] +
                                         b1 * self.u_vars[(idx_h, 1)], txt_dyn_tmp4)

    def set_lane_constraints(self, rmin_down, rmin_up):
        """
        Sets lane constraints.
        :param rmin_down: robustness slackness lane-down (dim = N x 1)
        :param rmin_up: robustness slackness lane-up (dim = N x 1)
        """
        # list_h_sel = self.list_h_relaxed
        list_h_sel = self.list_h
        a_d = -math.tan(self.rad_l)
        b_d = {i: rmin_down[i] - math.tan(self.rad_l) * self.cp_l_d[0] + self.cp_l_d[1] for i in list_h_sel}

        self.opt_model.addConstrs((a_d * self.x_vars[(i, 0)] + self.x_vars[(i, 1)] >= b_d[i] for i in list_h_sel),
                                  name='ld')

        a_u = math.tan(self.rad_l)
        # b_u = float(rmin_up + math.tan(self.rad_l) * self.cp_l_u[0] - self.cp_l_u[1])
        b_u = {i: rmin_up[i] + math.tan(self.rad_l) * self.cp_l_u[0] - self.cp_l_u[1] for i in list_h_sel}
        self.opt_model.addConstrs((a_u * self.x_vars[(i, 0)] - self.x_vars[(i, 1)] >= b_u[i] for i in list_h_sel),
                                  name='lu')

        # for nidx_h in range(0, self.h):
        #     y_down = math.tan(self.rad_l) * (x_vars[(nidx_h, 0)] - self.cp_l_d[0]) + self.cp_l_d[1]
        #     self.opt_model.addConstr(x_vars[(nidx_h, 1)] - y_down >= rmin_down, name="l_d{:d}".format(nidx_h))
        #
        #     y_up = math.tan(self.rad_l) * (x_vars[(nidx_h, 0)] - self.cp_l_u[0]) + self.cp_l_u[1]
        #     self.opt_model.addConstr(y_up - x_vars[(nidx_h, 1)] >= rmin_up, name="l_u{:d}".format(nidx_h))

    def set_collision_constraints(self, x_vars, z_vars, r_vars, rmin, traj_in, size_in, txt_label, param_x1=1.0,
                                  param_x2=0.125, param_x3=1.25, param_y1=0.1, param_y2=0.125, param_y3=0.1,
                                  param_y4=0.2, lanewidth=3.28):
        """
        Set collision constraints.
        :param x_vars: state optimization-variables
        :param z_vars: boolean optimization-variables
        :param r_vars: robustness optimization-variables
        :param rmin: robustness slackness (dim = H x 1)
        :param traj_in: trajectory (other vehicle) (dim = (H+1) x 2)
        :param size_in: size (other vehicle) (dim = (H+1) x 2)
        :param txt_label: label text (string)
        :param param_x1: parameter (float)
        :param param_x2: parameter (float)
        :param param_x3: parameter (float)
        :param param_y1: parameter (float)
        :param param_y2: parameter (float)
        :param param_y3: parameter (float)
        :param param_y4: parameter (float)
        :param lanewidth: lanewidth (float)
        """
        # list_h_sel = self.list_h_relaxed
        list_h_sel = self.list_h
        diff_traj_tmp = traj_in[range(1, self.h+1), 0:2] - traj_in[range(0, self.h), 0:2]
        dist_traj_tmp = np.sqrt(np.sum(diff_traj_tmp * diff_traj_tmp, axis=1))

        dist_traj = 0
        for idx_h in list_h_sel:
            x_oc_sel = traj_in[idx_h + 1, :]

            # Modify size w.r.t. distance
            dist_traj = min(dist_traj + dist_traj_tmp[idx_h] * lanewidth / 3.28, 1000.0)
            mod_size_x = min(param_x1 * (math.exp(param_x2 * dist_traj) - 1.0), param_x3)
            mod_size_y = param_y1 + min(param_y2 * (math.exp(param_y3 * dist_traj) - 1.0), param_y4)
            rx_oc = (size_in[idx_h + 1, 0] + self.rx) / 2.0 + mod_size_x
            ry_oc = (size_in[idx_h + 1, 1] + self.ry) / 2.0 + mod_size_y

            r_b_oc0, r_b_oc1 = float(-x_oc_sel[0] - rx_oc), float(x_oc_sel[0] - rx_oc)
            r_b_oc2, r_b_oc3 = float(-x_oc_sel[1] - ry_oc), float(x_oc_sel[1] - ry_oc)

            self.opt_model.addConstr(z_vars[(idx_h, 0)] + z_vars[(idx_h, 1)] + z_vars[(idx_h, 2)] +
                                     z_vars[(idx_h, 3)] == 1, name="{:s}_z{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] - x_vars[(idx_h, 0)] >= r_b_oc0, name="{:s}_0{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(r_vars[idx_h] + x_vars[(idx_h, 0)] >= r_b_oc1, name="{:s}_1{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(r_vars[idx_h] - x_vars[(idx_h, 1)] >= r_b_oc2, name="{:s}_2{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(r_vars[idx_h] + x_vars[(idx_h, 1)] >= r_b_oc3, name="{:s}_3{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] - x_vars[(idx_h, 0)] - self.M * z_vars[(idx_h, 0)] - r_b_oc0 +
                                     self.M >= 0.0, name="{:s}_4l{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(-r_vars[idx_h] + x_vars[(idx_h, 0)] - self.M * z_vars[(idx_h, 0)] + r_b_oc0 +
                                     self.M >= 0.0, name="{:s}_4r{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] + x_vars[(idx_h, 0)] - self.M * z_vars[(idx_h, 1)] - r_b_oc1 +
                                     self.M >= 0.0, name="{:s}_5l{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(-r_vars[idx_h] - x_vars[(idx_h, 0)] - self.M * z_vars[(idx_h, 1)] + r_b_oc1 +
                                     self.M >= 0.0, name="{:s}_5r{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] - x_vars[(idx_h, 1)] - self.M * z_vars[(idx_h, 2)] - r_b_oc2 +
                                     self.M >= 0.0, name="{:s}_6l{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(-r_vars[idx_h] + x_vars[(idx_h, 1)] - self.M * z_vars[(idx_h, 2)] + r_b_oc2 +
                                     self.M >= 0.0, name="{:s}_6r{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] + x_vars[(idx_h, 1)] - self.M * z_vars[(idx_h, 3)] - r_b_oc3 +
                                     self.M >= 0.0, name="{:s}_7l{:d}".format(txt_label, idx_h))
            self.opt_model.addConstr(-r_vars[idx_h] - x_vars[(idx_h, 1)] - self.M * z_vars[(idx_h, 3)] + r_b_oc3 +
                                     self.M >= 0.0, name="{:s}_7r{:d}".format(txt_label, idx_h))

            self.opt_model.addConstr(r_vars[idx_h] >= float(rmin[idx_h]), name="{:s}_8{:d}".format(txt_label, idx_h))

    def set_speed_constraints(self, rmin_speed):
        """
        Sets speed constraints.
        :param rmin_speed: robustness slackness speed (dim = H x 1)
        """
        # list_h_sel = self.list_h_relaxed
        list_h_sel = self.list_h
        for idx_h in list_h_sel:
            self.opt_model.addConstr(self.v_th - self.x_vars[(idx_h, 3)] >= rmin_speed[idx_h], name="speed_0{:d}".format(idx_h))

    def set_until_constraint(self, rmin_until):
        """
        Sets until-logic constraint.
        :param rmin_until: robustness slackness until (float)
        """
        # Until
        for nidx_t in range(self.until_t_a, self.until_t_b):
            self.set_until_constraint_ubuntil(nidx_t)

        # Always
        self.set_until_constraint_alw()

        # Eventually
        self.set_until_constraint_ev()

        # and1 (always + eventually)
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_alw, name="until_and1_r0")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_ev, name="until_and1_r1")

        self.opt_model.addConstr(self.z_until_and1[0] + self.z_until_and1[1] == 1, name="until_and1_M")
        self.opt_model.addConstr(self.r_until_alw - (1 - self.z_until_and1[0]) * self.M <= self.r_until_and1,
                                 name="until_and1_M0")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_alw + (1 - self.z_until_and1[0]) * self.M,
                                 name="until_and1_M1")
        self.opt_model.addConstr(self.r_until_ev - (1 - self.z_until_and1[1]) * self.M <= self.r_until_and1,
                                 name="until_and1_M2")
        self.opt_model.addConstr(self.r_until_and1 <= self.r_until_ev + (1 - self.z_until_and1[1]) * self.M,
                                 name="until_and1_M3")

        # and2 (+ until)
        self.opt_model.addConstr(self.r_until <= self.r_until_and1, name="until_and2_r0")
        self.opt_model.addConstr(self.r_until <= self.r_until_or_hist[0], name="until_and2_r1")

        self.opt_model.addConstr(self.z_until_and2[0] + self.z_until_and2[1] == 1, name="until_and2_M")
        self.opt_model.addConstr(self.r_until_and1 - (1 - self.z_until_and2[0]) * self.M <= self.r_until,
                                 name="until_and2_M0")
        self.opt_model.addConstr(self.r_until <= self.r_until_and1 + (1 - self.z_until_and2[0]) * self.M,
                                 name="until_and2_M1")
        self.opt_model.addConstr(self.r_until_or_hist[0] - (1 - self.z_until_and2[1]) * self.M <= self.r_until,
                                 name="until_and2_M2")
        self.opt_model.addConstr(self.r_until <= self.r_until_or_hist[0] + (1 - self.z_until_and2[1]) * self.M,
                                 name="until_and2_M3")

        self.opt_model.addConstr(self.r_until >= rmin_until, name="until_final")

    def set_until_constraint_ubuntil(self, idx_t):
        """
        Sets unbound until-logic constraint.
        :param idx_t: index of xvars (int)
        """
        idx_t_m = idx_t - self.until_t_a  # Modified index

        r_phi1 = self.until_v_th - self.x_vars[(idx_t, 3)]
        r_phi2 = self.x_vars[(idx_t, 0)] - self.traj_ov_cf[idx_t + 1, 0] + \
                 (self.rx + self.size_ov_cf[idx_t + 1, 0]) / 2.0 + self.until_d_th

        if idx_t == (self.until_t_b - 1):
            r_phi2_next = self.x_vars[(idx_t + 1, 0)] - self.traj_ov_cf[idx_t + 2, 0] + \
                          (self.rx + self.size_ov_cf[idx_t + 2, 0]) / 2.0 + self.until_d_th

            # (and)
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1, name="until_h_and_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi2_next, name="until_h_and_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_and_hist[(idx_t_m, 0)] + self.z_until_and_hist[(idx_t_m, 1)] == 1,
                                     name="until_h_and_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_and_hist[idx_t_m], name="until_h_and_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1 +
                                     (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M,
                                     name="until_h_and_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2_next - (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_and_hist[idx_t_m], name="until_h_and_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi2_next +
                                     (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M,
                                     name="until_h_and_M3{:d}".format(idx_t_m))

            # (or)
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= r_phi2,
                                     name="until_h_or_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= self.r_until_and_hist[idx_t_m],
                                     name="until_h_or_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_or_hist[(idx_t_m, 0)] + self.z_until_or_hist[(idx_t_m, 1)] == 1,
                                     name="until_h_or_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_or_hist[idx_t_m], name="until_h_or_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(
                self.r_until_or_hist[idx_t_m] <= r_phi2 + (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M,
                name="until_h_or_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] - (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_or_hist[idx_t_m], name="until_h_or_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= self.r_until_and_hist[idx_t_m] +
                                     (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M,
                                     name="until_h_or_M3{:d}".format(idx_t_m))
        else:
            # (and)
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1, name="until_h_and_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= self.r_until_or_hist[idx_t_m + 1],
                                     name="until_h_and_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_and_hist[(idx_t_m, 0)] + self.z_until_and_hist[(idx_t_m, 1)] == 1,
                                     name="until_h_and_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_and_hist[idx_t_m], name="until_h_and_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= r_phi1 + (1 - self.z_until_and_hist[(idx_t_m, 0)]) * self.M,
                                     name="until_h_and_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m + 1] - (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_and_hist[idx_t_m], name="until_h_and_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] <= self.r_until_or_hist[idx_t_m + 1] +
                                     (1 - self.z_until_and_hist[(idx_t_m, 1)]) * self.M, name="until_h_and_M3{:d}".format(idx_t_m))

            # (or)
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= r_phi2, name="until_h_or_r0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] >= self.r_until_and_hist[idx_t_m],
                                     name="until_h_or_r1{:d}".format(idx_t_m))

            self.opt_model.addConstr(self.z_until_or_hist[(idx_t_m, 0)] + self.z_until_or_hist[(idx_t_m, 1)] == 1,
                                     name="until_h_or_M{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M <=
                                     self.r_until_or_hist[idx_t_m], name="until_h_or_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= r_phi2 + (1 - self.z_until_or_hist[(idx_t_m, 0)]) * self.M,
                                     name="until_h_or_M1{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_and_hist[idx_t_m] - (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M
                                     <= self.r_until_or_hist[idx_t_m], name="until_h_or_M2{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_or_hist[idx_t_m] <= self.r_until_and_hist[idx_t_m] +
                                     (1 - self.z_until_or_hist[(idx_t_m, 1)]) * self.M,
                                     name="until_h_or_M3{:d}".format(idx_t_m))

    def set_until_constraint_alw(self):
        """
        Sets until constraint (always).
        """
        self.opt_model.addConstr(sum(self.z_until_alw[i] for i in range(0, self.until_t_a - self.until_t_s)) == 1, name="until_alw_M")
        for idx_t in range(self.until_t_s, self.until_t_a):
            idx_t_m = idx_t - self.until_t_s  # Modified index
            r_phi1 = self.until_v_th - self.x_vars[(idx_t, 3)]

            self.opt_model.addConstr(self.r_until_alw <= r_phi1, name="until_alw_r{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi1 - (1 - self.z_until_alw[idx_t_m]) * self.M <= self.r_until_alw,
                                     name="until_alw_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_alw <= r_phi1 + (1 - self.z_until_alw[idx_t_m]) * self.M,
                                     name="until_alw_M1{:d}".format(idx_t_m))

    def set_until_constraint_ev(self):
        """
        Sets until constraint (eventually).
        """
        self.opt_model.addConstr(sum(self.z_until_ev[i] for i in range(0, self.until_t_b + 1 - self.until_t_a)) == 1, name="until_ev_M")

        for idx_t in range(self.until_t_a, self.until_t_b + 1):
            idx_t_m = idx_t - self.until_t_a  # Modified index
            r_phi2 = self.x_vars[(idx_t, 0)] - self.traj_ov_cf[idx_t + 1, 0] + (self.rx + self.size_ov_cf[idx_t + 1, 0]) / 2.0 + self.until_d_th

            self.opt_model.addConstr(self.r_until_ev >= r_phi2, name="until_ev_r{:d}".format(idx_t_m))
            self.opt_model.addConstr(r_phi2 - (1 - self.z_until_ev[idx_t_m]) * self.M <= self.r_until_ev,
                                     name="until_ev_M0{:d}".format(idx_t_m))
            self.opt_model.addConstr(self.r_until_ev <= r_phi2 + (1 - self.z_until_ev[idx_t_m]) * self.M,
                                     name="until_ev_M1{:d}".format(idx_t_m))

    # OTHERS ----------------------------------------------------------------------------------------------------------#
    def compute_robustness_slackness_init(self, rmin_l_d_in, rmin_l_u_in, rmin_c_cf_in, rmin_c_rest_in, rmin_speed_in):
        """
        Computes robustness (for initial state).
        :param rmin_l_d_in: robustness slackness lane-down (float)
        :param rmin_l_u_in: robustness slackness lane-up (float)
        :param rmin_c_cf_in: robustness slackness collision-centerfront (float)
        :param rmin_c_rest_in: robustness slackness collision-rest (float)
        :param rmin_speed_in: robustness slackness collision-speed (float)
        """

        idx_h_oc = np.arange(0, 1)
        r_l_down, r_l_up, r_c_cf, r_c_rest, r_c_rest_array, r_speed = \
            compute_robustness_part(self.xinit_conv, [self.rx, self.ry], self.cp_l_d, self.cp_l_u, self.rad_l,
                                    self.traj_ov_cf, self.size_ov_cf, self.traj_ov, self.size_ov, idx_h_oc, self.v_th)

        self.r_l_down_init = r_l_down
        self.r_l_up_init = r_l_up
        self.r_c_cf_init = r_c_cf
        self.r_c_rest_array_init = r_c_rest_array

        idx_relax = np.arange(0, self.h_relax)

        # Center-front vehicle ------------------------------------------------------------------------------#
        id_cf = self.id_ov_near[4]
        traj_ov_cr = self.traj_ov[4]
        size_ov_cr = self.size_ov[4]
        mindist2cf_0 = 0.5 * (self.rx + self.size_ov_cf[0, 0])
        mindist2cf_1 = 0.5 * (self.rx + self.size_ov_cf[-1, 0])
        mindist2cr_1 = 0.5 * (self.rx + size_ov_cr[-1, 0])
        x_cf_1 = self.traj_ov_cf[-1, 0] - 1.5 * mindist2cf_1
        x_cr_1 = traj_ov_cr[-1, 0] + 1.5 * mindist2cr_1

        # Modify 'rmin_c_cf_in'
        if id_cf > 0:
            # Consider future position of other vehicles (cf, rf)
            d_cf_rf = x_cf_1 - x_cr_1
            rmin_c_cf_max = min(x_cf_1, 0.6 * d_cf_rf)

            if rmin_c_cf_max > 0:
                rmin_c_cf_in_ = min(rmin_c_cf_max, rmin_c_cf_in)
            else:
                rmin_c_cf_in_ = rmin_c_cf_in

            rmin_c_cf_in = max(rmin_c_cf_in_, 0)

            # When center-front vehicle is too close
            if (self.traj_ov_cf[0, 0] < (2.0 * mindist2cf_0)) and (id_cf > 0):
                rmin_c_cf_in = max(r_c_cf - 0.5 * mindist2cf_0, 0)

        self.rmin_c_cf_in = rmin_c_cf_in

        # Rule 1-2: lane-observation (down, up) -------------------------------------------------------------#
        rmin_l_d = rmin_l_d_in * np.ones((self.h,), dtype=np.float32)
        rmin_l_u = rmin_l_u_in * np.ones((self.h,), dtype=np.float32)

        if self.h_relax > 0:
            rmin_l_d[idx_relax] = min(r_l_down - 0.25, rmin_l_d_in)
            rmin_l_u[idx_relax] = min(r_l_up - 0.25, rmin_l_u_in)

        # Rule 3: collision (front)  ------------------------------------------------------------------------#
        rmin_c_cf = self.rmin_c_cf_in * np.ones((self.h,), dtype=np.float32)
        if self.h_relax > 0:
            rmin_c_cf[idx_relax] = min(max(r_c_cf - mindist2cf_0, 0), rmin_c_cf_in)
            # rmin_c_cf[idx_relax] = 0

        # Rule 4: collision (rest)  -------------------------------------------------------------------------#
        num_ov_rest = len(self.traj_ov)
        rmin_c_rest = []
        for nidx_oc in range(0, num_ov_rest):
            rmin_c_rest_tmp = rmin_c_rest_in * np.ones((self.h,), dtype=np.float32)
            if self.h_relax > 0:
                # rmin_c_rest_tmp[idx_relax] = min(max(r_c_rest_array[nidx_oc] - 10, 0), rmin_c_rest_in)
                rmin_c_rest_tmp[idx_relax] = 0

            rmin_c_rest.append(rmin_c_rest_tmp)

        # Rule 5: speed-limit  ------------------------------------------------------------------------------#
        rmin_c_speed = rmin_speed_in * np.ones((self.h,), dtype=np.float32)
        if self.h_relax > 0:
            rmin_c_speed[idx_relax] = min(r_speed - 1.25, rmin_speed_in)
            # rmin_c_speed[idx_relax] = -20

        return rmin_l_d, rmin_l_u, rmin_c_cf, rmin_c_rest, rmin_c_speed

    def compute_cost(self, traj, u):
        """
        Computes cost.
        :param traj: ego-vehicle trajectory (ndarray, dim = H x 4)
        :param u: ego-vehicle control (ndarray, dim = H x 2)
        """
        traj = make_numpy_array(traj, keep_1dim=False)
        u = make_numpy_array(u, keep_1dim=False)

        len_traj_in = traj.shape[0]
        len_u_in = u.shape[0]

        cost_xend = np.zeros((self.dim_x,), dtype=np.float32)
        for nidx_d in range(0, self.dim_x):
            cost_xend[nidx_d] = (traj[len_traj_in - 1, nidx_d] - float(self.xgoal_conv[nidx_d])) * \
                                (traj[len_traj_in - 1, nidx_d] - float(self.xgoal_conv[nidx_d])) * \
                                float(self.c_end[nidx_d])

        cost_u = np.zeros((self.dim_u,), dtype=np.float32)
        for nidx_h in range(0, len_u_in):
            for nidx_d in range(0, self.dim_u):
                cost_u[nidx_d] += (u[nidx_h, nidx_d]) * (u[(nidx_h, nidx_d)]) * float(self.c_u[nidx_d])

        return cost_xend, cost_u

    def get_modified_size_linear(self, size, heading, w=0.4):
        """
        Gets modified size for linearization.
        :param size: dx dy (ndarray, dim = 2)
        :param heading: heading (float)
        :param w: weight (float)
        """
        if abs(math.cos(heading)) <= 0.125:
            size_out = np.flipud(size)
        elif abs(math.sin(heading)) <= 0.125:
            size_out = size
        elif abs(math.cos(heading)) < 1 / math.sqrt(2):
            size_x = w * abs(size[0] * math.cos(heading)) + abs(size[1] * math.sin(heading))
            size_y = abs(size[0] * math.sin(heading)) + w * abs(size[1] * math.cos(heading))
            size_out = np.array([size_x, size_y], dtype=np.float32)
        else:
            size_x = abs(size[0] * math.cos(heading)) + w * abs(size[1] * math.sin(heading))
            size_y = w * abs(size[0] * math.sin(heading)) + abs(size[1] * math.cos(heading))
            size_out = np.array([size_x, size_y], dtype=np.float32)

        # size_out = size_in
        return size_out

    def update_history(self):
        """
        Updates history.
        """
        self.cnt_hist_ev = self.cnt_hist_ev + 1

        self.x_hist_ev[self.cnt_hist_ev - 1, :] = self.xinit
        self.xgoal_hist_ev[self.cnt_hist_ev - 1, :] = self.xgoal[0:2]
        self.cpd_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_d_rec
        self.cpu_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_u_rec

        self.x_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.xinit_conv
        self.xgoal_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.xgoal_conv[0:2]
        self.cpd_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_d
        self.cpu_conv_hist_ev[self.cnt_hist_ev - 1, :] = self.cp_l_u

    def print_r_until(self):
        """
        Prints r_until optimization results.
        """
        r_until_out = self.r_until.X
        print("r_until: {:.4f}".format(r_until_out))

        r_until_and1_out = self.r_until_and1.X
        print("r_until_and1: {:.4f}".format(r_until_and1_out))

        r_until_or_hist_out = []
        for nidx_d in range(0, len(self.r_until_or_hist)):
            r_until_or_hist_out.append(self.r_until_or_hist[nidx_d].X)
        print("r_until_or_hist: (min) {:.4f}, (max) {:.4f}".format(min(r_until_or_hist_out), max(r_until_or_hist_out)))

        r_until_and_hist_out = []
        for nidx_d in range(0, len(self.r_until_and_hist)):
            r_until_and_hist_out.append(self.r_until_and_hist[nidx_d].X)
        print("r_until_and_hist: (min) {:.4f}, (max) {:.4f}".format(min(r_until_and_hist_out), max(r_until_and_hist_out)))
