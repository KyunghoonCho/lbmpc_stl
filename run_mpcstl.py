# MODEL PREDICTIVE CONTROL UNDER STL-CONSTRAINTS (MPCSTL)
#   - Model predictive control unser STL constraints
#   - Reference:
#       [1] Kyunghoon Cho and Songhwai Oh,
#       "Learning-Based Model Predictive Control under Signal Temporal Logic Specifications,"
#       in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), May 2018.
#
#   - STL rules:
#       1: Lane-observation (down, right)
#       2: Lane-observation (up, left)
#       3: Collision (front)
#       4: Collision (others)
#       5: Speed-limit
#       6: Slow-down
#
#   - System Dynamic: 4-state unicycle dynamics
#       state: [x, y, theta, v]
#       control: [w, a]
#           dot(x) = v * cos(theta)
#           dot(y) = v * sin(theta)
#           dot(theta) = v * kappa_1 * w
#           dot(v) = kappa_2 * a
#
#   - Pycharm
#       Run/debug configuration:
#           LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/gurobi901/linux64/lib

from __future__ import print_function

import argparse
import os

from core.SimControl import *
from core.SimScreen import *
from core.SimTrack import *
from MPCSTL import *


def main(params):

    model_name= "mpcstl"
    np.random.seed(0)  # Set random seed

    # SET KEY PARAMETERS ----------------------------------------------------------------------------------------------#
    MODE_INIT, VIEW_SCREEN, MODE_SCREEN = params['MODE_INIT'], params['VIEW_SCREEN'], params['MODE_SCREEN']
    PLOT_BK, PLOT_DEBUG, SAVE_RESULT = params['PLOT_BK'], params['PLOT_DEBUG'], params['SAVE_RESULT']
    VERBOSE = params['VERBOSE']

    # Environment parameter (track-name, id of ego-vehicle)
    trackname, id_ev = params['trackname'], params['v_id']

    # Set robustness slackness
    rmin_l_r, rmin_l_l, rmin_obs_cf, rmin_obs_o, rmin_vel, rmin_until = 0, 0, 0, 0, 0, -100

    # SET PARAMETERS --------------------------------------------------------------------------------------------------#
    # System parameter
    dt = 0.1  # Time-step
    rx_ev, ry_ev = 4.2, 1.9  # Size of ego-vehicle
    kappa = [0.5, 1]  # Dynamic parameters
    v_ref = 7.0  # Reference (linear) velocity
    v_range = [0, 30]  # Range of (linear) velocity

    # MIP parameter
    u_range = np.array([[-math.pi * 0.5, +math.pi * 0.5], [-500, +500]], dtype=np.float32)
    horizon_mpc = 18  # Horizon of trajectory (16)
    horizon_relax = 4  # Horizon of stl-constraint relaxation
    c_end_0 = [1, 5, 0.1, 0.1]  # Cost parameters (end-point-1)
    c_end_1 = [1, 20, 0.1, 0.1]  # Cost parameters (end-point-1)
    c_u_0 = [2e2, 0.1]  # Cost parameters (control-1) ([2e2, 0.1])
    c_u_1 = [1e1, 0.1]  # Cost parameters (control-2) ([2e2, 0.1])
    dth_c_end = 3.0 * rx_ev  # Distance threshold (c_end)
    rth_c_u = 10.0 / 180.0 * np.pi  # Radiance threshold (c_u)
    dist_goal_ahead = horizon_mpc * v_ref * 0.25  # Distance to the goal state

    # Lane-observation parameter
    min_l_width = 1.25 * ry_ev  # Minimum space where the vehicle can exist (w.r.t. lane-observation rules)

    # Velocity threshold
    v_th = 20

    # Until-logic parameter
    dw_use_until = 4.0  # Distance weight (use_until)
    dsth_use_until = 0.33  # (Scaled) distance threshold (use_until)
    rth_use_until = 12.5 / 180 * np.pi  # Radiance threshold (use_until)
    until_t_s, until_t_a, until_t_b = int(horizon_mpc / 4), int(horizon_mpc / 2), (horizon_mpc - 1)
    until_v_th, until_d_th = 12, 6  # 12, 6
    until_lanewidth = 3.6

    # Naive-control parameter
    max_cnt_consec_naive = 3  # Maximum number of consecutive naive-trials
    u_w_set = np.linspace(-math.pi * 0.1, +math.pi * 0.1, num=5)  # Set of angular-velocities
    u_a_set = np.linspace(-20, +20, num=15)  # Set of accelerations
    u_set = get_two_united_set(u_w_set, u_a_set)  # Set of control

    # Goal setting parameter
    g_alpha = 0.1  # Toggle goal distance-threshold weight

    # SET TRACK -------------------------------------------------------------------------------------------------------#
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    trackname_split = trackname.split("_")
    trackname_track, trackname_id = trackname_split[0], trackname_split[1]
    if trackname_track == "us101" or trackname_track == "i80":
        vehicle_data_filename = "./data_vehicle/dv_ngsim_{:s}_{:s}.npy".format(trackname_track.lower(), trackname_id)
        is_track_simple = 0
    elif "highd" in trackname_track:
        vehicle_data_filename = "./data_vehicle/dv_highD_{:s}.npy".format(trackname_id)
        is_track_simple = 1
    else:
        vehicle_data_filename = "./data_vehicle/dv_ngsim_us101_1.npy"
        is_track_simple = 0

    # Initial indexes array for seg & lane
    if trackname_track == "us101":
        seg_init_list, lane_init_list = [0], [0, 1, 2, 3]
    elif trackname_track == "i80":
        seg_init_list, lane_init_list = [0], [0, 1, 2]
    elif "highd" in trackname_track:
        seg_init_list, lane_init_list = [0], np.arange(1, sim_track.num_lane[0], 2)
    else:
        seg_init_list, lane_init_list = [0, 1], [0, 1, 2, 3, 4]

    data_v_in = np.load(vehicle_data_filename)
    t_min, t_max = int(np.amin(data_v_in[:, 0])), int(np.amax(data_v_in[:, 0]))
    min_vel, max_vel, mean_vel = min(data_v_in[:, 4]), max(data_v_in[:, 4]), np.mean(data_v_in[:, 4])
    print("[SET-ENV] track-name: {:s}, vehicle-id: {:d}".format(trackname, id_ev))
    print("min_vel: {:.2f}, max_vel: {:.2f}, mean_vel: {:.2f}".format(min_vel, max_vel, mean_vel))

    # SET CONTROL -----------------------------------------------------------------------------------------------------#
    id_ev, t_init, t_horizon, data_ev_init, data_ev, data_ov = \
        set_initstate_control(id_ev, v_ref, ry_ev, rx_ev, seg_init_list, lane_init_list, data_v_in, t_min, t_max,
                              MODE_INIT, sim_track, is_track_simple=is_track_simple)

    sim_control = SimControl(sim_track, dt, rx_ev, ry_ev, kappa, v_ref, v_range)
    sim_control.update_initstate(data_ev_init[1], data_ev_init[2], data_ev_init[3], data_ev_init[4])
    sim_control.update_state(data_ev_init[1], data_ev_init[2], data_ev_init[3], data_ev_init[4])

    sim_mpcstl = MPCSTL(dt, rx_ev, ry_ev, kappa, horizon_mpc, horizon_relax, v_ref, v_range,
                        u_range[:, 0], u_range[:, 1], dist_goal_ahead,
                        c_end_0, c_end_1, c_u_0, c_u_1, dth_c_end, rth_c_u, v_th,
                        dw_use_until, dsth_use_until, rth_use_until,
                        until_t_s, until_t_a, until_t_b, until_v_th, until_d_th, until_lanewidth, VERBOSE)

    # SET SCREEN ------------------------------------------------------------------------------------------------------#
    if VIEW_SCREEN == 1:
        height_add = 0 if PLOT_DEBUG == 0 else 300
        sim_screen = SimScreen(sim_track, MODE_SCREEN, PLOT_BK, height_add=height_add)

        # SUB-SCREEN 2
        if PLOT_DEBUG >= 1:
            sim_screen.set_pnts_range_sub([0, 0], [80, 12.5], sub_num=2)
            sim_screen.set_screen_height_sub(sim_screen.screen_size[1] * (1 / 7.5), sub_num=2)
            sim_screen.set_screen_alpha_sub(sub_num=2)
    else:
        sim_screen = []

    # SET DIRECTORY ---------------------------------------------------------------------------------------------------#
    if SAVE_RESULT >= 1:
        if id_ev < 0:
            directory2save = "./result/{:s}/{:s}_idneg".format(model_name, trackname)
        else:
            directory2save = "./result/{:s}/{:s}_id{:d}".format(model_name, trackname, id_ev)
        if not os.path.exists(directory2save):
            os.makedirs(directory2save)

        if VIEW_SCREEN == 1 and SAVE_RESULT == 2:
            directory2save_pic = "{:s}/pic".format(directory2save)
            if not os.path.exists(directory2save_pic):
                os.makedirs(directory2save_pic)

    # MAIN: RUN SIM ---------------------------------------------------------------------------------------------------#
    # Set repository
    data_ev_t_exp, data_ov_t_exp, s_ev_t_exp = [], [], []
    id_near_t_exp, id_rest_t_exp = [], []
    traj_ov_near_exp, size_ov_near_exp = [], []
    y_ev_exp, traj_ev_exp = [], []

    r_measured_hist = np.zeros((t_horizon + 1, 6), dtype=np.float32)
    cost_measured_hist = np.zeros((t_horizon + 1, 7), dtype=np.float32)  # 1 + dim_x + dim_u
    error_hist = np.zeros((t_horizon + 1, 3), dtype=np.int32)  # num_trial, is_error, is_error_until

    # Set initial goal-info
    seg_goal, lane_goal = 0, 0

    # Set initial outside-track-info
    is_outside = False  # Whether ego-vehicle is outside of track

    # Set initial collision-info
    is_collision = False  # Whether ego-vehicle is in collision

    # Set initial goal-reach-info
    is_reachgoal = False  # Whether ego-vehicle reaches goal points

    t_cur, cnt_naive = 0, 0
    print("[SIMULATION START]--------------------------------------------------")
    for t_cur in range(t_init, t_init + t_horizon + 1):
        # STEP1: GET CURRENT INFO -------------------------------------------------------------------------------------#
        # Select current vehicle data (w.r.t time)
        s_ev_t = [sim_control.x_ego, sim_control.y_ego, sim_control.theta_ego, sim_control.v_ego]
        data_ev_t, idx_ev_t = select_data_t(data_ev, t_cur)
        data_ov_t, idx_ov_t = select_data_t(data_ov, t_cur)
        idx_ev_t = idx_ev_t[0]

        seg_ev, lane_ev = int(data_ev_t[7]), int(data_ev_t[8])

        # Get feature
        dict_info_t = get_info_t(t_cur, id_ev, data_ev, data_ev_t, data_ov_t, [], 0, horizon_mpc, [], [],
                                 2, sim_track, use_intp=0)
        dist_cf = dict_info_t['dist_cf']
        id_near, id_rest = dict_info_t['id_near'], dict_info_t['id_rest']
        lanewidth, lanedev_rad = dict_info_t['lanewidth'], dict_info_t['lanedev_rad']
        lanedev_dist_scaled = dict_info_t['lanedev_dist_scaled']
        pnt_center, rad_center = dict_info_t['pnt_center'], dict_info_t['rad_center']
        pnt_left, pnt_right = dict_info_t['pnt_left'], dict_info_t['pnt_right']

        # Get near other vehicles
        data_ov_near_t, data_ov_near_t_list = select_data_near(data_v_in, id_near)
        traj_ov_near, size_ov_near = get_vehicle_traj_near(data_ov_near_t_list, id_near, t_cur, horizon_mpc, handle_remain=1)
        _, traj_ov_near_sq, size_ov_near_sq = get_vehicle_traj(data_ov_near_t, t_cur, horizon_mpc, handle_remain=1)

        # (CHECK) outside
        if seg_ev == -1 or lane_ev == -1:
            is_outside = True
            break

        # (CHECK) collision
        is_collision = check_collision_t(data_ev_t, data_ov_near_t, t_cur)
        if is_collision:
            break

        # (CHECK) reach-goal
        is_reachgoal = check_reachgoal(s_ev_t[0:2], sim_track.pnts_goal, d_th=0.5*lanewidth)
        if is_reachgoal:
            break

        # STEP2: SET GOAL STATE ---------------------------------------------------------------------------------------#
        lane_goal = lane_ev if t_cur == t_init else lane_goal
        _, lane_goal, goal_type_txt, state_array_goal, state_goal = \
            sim_mpcstl.get_goal_state(False, 1, 1, 0.01, lane_goal, s_ev_t, seg_ev, lane_ev, data_ov_t,
                                      traj_ov_near[4], size_ov_near[4], sim_control, trackname)

        # STEP3: SET ROBUSTNESS SLACKNESS -----------------------------------------------------------------------------#
        # Decision on whether to use until-logic
        use_until = sim_mpcstl.set_use_until(id_near, dist_cf, lanewidth, lanedev_dist_scaled, lanedev_rad, lane_ev,
                                             lane_goal)

        # STEP4: FIND CONTROL -----------------------------------------------------------------------------------------#
        cp2rotate_mpc, theta2rotate_mpc = pnt_center, rad_center
        sim_mpcstl.set_control_cost_mode(dist_cf, lanedev_rad)
        sim_mpcstl.convert_state(s_ev_t, state_goal, cp2rotate_mpc, theta2rotate_mpc)
        sim_mpcstl.get_lane_constraints(pnt_right, pnt_left, rad_center, 0.0, cp2rotate_mpc, theta2rotate_mpc)

        sim_mpcstl.get_collision_constraints(traj_ov_near, size_ov_near, id_near, cp2rotate_mpc,
                                             theta2rotate_mpc, lanewidth)
        sim_mpcstl.update_history()

        if VERBOSE >= 1:
            if VERBOSE == 2:
                print("[t:{:d}] xinit_conv: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(
                    t_cur, sim_mpcstl.xinit_conv[0], sim_mpcstl.xinit_conv[1], sim_mpcstl.xinit_conv[2],
                    sim_mpcstl.xinit_conv[3]))

            print("{:<16s} (lw: {:3.2f}, ldev_r: {:3.2f}, ldev_d: {:3.2f}), (goal: {:<2s}, c: {:d}{:d}, until: {:d})".
                  format("[FEATURE]", lanewidth, lanedev_rad, lanedev_dist_scaled, goal_type_txt,
                         sim_mpcstl.mode_cend, sim_mpcstl.mode_cu, use_until))
        print("{:<16s} {:6.3f}, {:6.3f}, {:7.3f}, {:7.2f}, {:5.2f}, {:5.2f}".
              format("[R_MIN]", rmin_l_r, rmin_l_l, sim_mpcstl.rmin_c_cf_in, rmin_obs_o, rmin_vel, rmin_until))

        # Solve MPC
        x_out, u_out, x_out_revert, num_trial, is_error, is_error_until = sim_mpcstl.control_by_mpc(
            rmin_l_r, rmin_l_l, rmin_obs_cf, 0, rmin_vel, rmin_until, cp2rotate_mpc, theta2rotate_mpc, use_until,
            lw=lanewidth)
        error_hist[t_cur - t_init, :] = [num_trial, is_error, is_error_until]

        if is_error == 0:
            traj_sel_ev = x_out_revert

            idx_h_ev, idx_h_ov = np.arange(1, sim_mpcstl.h + 1), np.arange(1, sim_mpcstl.h + 1)
            r_l_down_o, r_l_up_o, r_c_cf_o, r_c_rest_o, r_c_rest_array_o, r_speed_o, r_until_o = \
                compute_robustness(x_out[idx_h_ev, 0:3], x_out[idx_h_ev, 3], [sim_mpcstl.rx, sim_mpcstl.ry],
                                   sim_mpcstl.cp_l_d, sim_mpcstl.cp_l_u, sim_mpcstl.rad_l, sim_mpcstl.traj_ov_cf,
                                   sim_mpcstl.size_ov_cf, sim_mpcstl.traj_ov, sim_mpcstl.size_ov, idx_h_ov,
                                   sim_mpcstl.v_th,
                                   sim_mpcstl.until_t_s, sim_mpcstl.until_t_a, sim_mpcstl.until_t_b,
                                   sim_mpcstl.until_v_th,
                                   sim_mpcstl.until_d_th)

            cost_xend, cost_u = sim_mpcstl.compute_cost(x_out, u_out)
            cost_all = np.sum(cost_xend) + sum(cost_u)

            print("{:<16s} {:6.3f}, {:6.3f}, {:7.3f}, {:7.3f}, {:5.3f}, {:5.3f}".
                  format("[R_MEASURED]", r_l_down_o, r_l_up_o, r_c_cf_o, r_c_rest_o, r_speed_o, r_until_o))

            if VERBOSE == 2:
                print("[MIP-CONTROL] xend_conv: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]".
                      format(t_cur, x_out[-1, 0], x_out[-1, 1], x_out[-1, 2], x_out[-1, 3]))
                print("[COST] cost_all: {:.3f}, cost_xend: [{:.3f}, {:.3f}, {:.3f}, {:.3f}], cost_u: [{:.3f}, {:.3f}]".
                      format(cost_all, cost_xend[0], cost_xend[1], cost_xend[2], cost_xend[3], cost_u[0], cost_u[1]))
            else:
                print("[t:{:d}] MIP-CONTROL, error_until:{:d}, r:[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]".
                      format(t_cur, is_error_until, r_l_down_o, r_l_up_o, r_c_cf_o, r_c_rest_o, r_speed_o,
                             r_until_o))

            cnt_naive = 0  # Reset count
        else:
            if cnt_naive < max_cnt_consec_naive:
                print("[t:{:d}] NAIVE-CONTROL".format(t_cur))
                traj_sel_ev, traj_array_ev, cost_array_ev, idx_traj_invalid_ev = \
                    sim_control.find_control_naive(u_set, horizon_mpc, data_ov_near_t, t_cur, state_goal[0:3])
            else:
                traj_sel_ev = np.array([s_ev_t, s_ev_t], dtype=np.float32)

            r_l_down_o, r_l_up_o, r_c_cf_o, r_c_rest_o, r_speed_o, r_until_o = 0, 0, 0, 0, 0, 0
            cost_xend = np.zeros((sim_mpcstl.dim_x,), dtype=np.float32)
            cost_u = np.zeros((sim_mpcstl.dim_u,), dtype=np.float32)
            cost_all = 0

            cnt_naive += 1  # Increase count

        # STEP5: UPDATE -----------------------------------------------------------------------------------------------#
        s_ev_next = traj_sel_ev[1, :]
        seg_ev_next_, lane_ev_next_ = get_index_seglane([s_ev_next[0], s_ev_next[1]], sim_track.pnts_poly_track)
        seg_ev_next, lane_ev_next = seg_ev_next_[0], lane_ev_next_[0]
        sim_control.update_state(s_ev_next[0], s_ev_next[1], s_ev_next[2], s_ev_next[3])
        # data_ev[idx_ev_cur + 1, 1:5] = [s_ev_next[0], s_ev_next[1], s_ev_next[2], s_ev_next[3]]
        data_ev[idx_ev_t + 1, 1:4] = [s_ev_next[0], s_ev_next[1], s_ev_next[2]]
        data_ev[idx_ev_t + 1, 4] = norm(s_ev_next[0:2] - s_ev_t[0:2]) / dt
        data_ev[idx_ev_t + 1, 7:9] = [seg_ev_next, lane_ev_next]

        sim_control.update_state(s_ev_next[0], s_ev_next[1], s_ev_next[2], s_ev_next[3])

        # Update repository
        data_ev_t_exp.append(data_ev_t)
        data_ov_t_exp.append(data_ov_t)
        s_ev_t_exp.append(s_ev_t)
        id_near_t_exp.append(id_near)
        id_rest_t_exp.append(id_rest)
        traj_ov_near_exp.append(traj_ov_near_sq)
        size_ov_near_exp.append(size_ov_near_sq)
        y_ev_exp.append(traj_sel_ev)
        traj_hist_ev = sim_mpcstl.x_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2]
        traj_ev_exp.append(traj_hist_ev)

        if is_error == 0:
            r_measured_hist[t_cur - t_init, :] = [r_l_down_o, r_l_up_o, r_c_cf_o, r_c_rest_o, r_speed_o, r_until_o]
            cost_measured_hist[t_cur - t_init, 0] = cost_all
            cost_measured_hist[t_cur - t_init, 1:] = np.concatenate((cost_xend, cost_u), axis=0)

        # STEP6: DRAW -------------------------------------------------------------------------------------------------#
        if VIEW_SCREEN == 1:
            # Draw track
            sim_screen.draw_basic(s_ev_t)

            v_arrow = sim_control.v_range[1]
            sim_screen.draw_ctrl_basic(data_ev_t, data_ov_t, id_ev, id_near, id_rest, traj_sel_ev, traj_hist_ev,
                                       traj_ov_near_sq, size_ov_near_sq, v_arrow)

            if PLOT_DEBUG >= 2:  # PLOT FOR DEBUGGING
                sim_screen.draw_traj(sim_mpcstl.cpd_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2],
                                     2.5, sim_mpcstl.hcolor_lane_d)
                sim_screen.draw_traj(sim_mpcstl.cpu_hist_ev[range(0, sim_mpcstl.cnt_hist_ev), 0:2],
                                     2.5, sim_mpcstl.hcolor_lane_u)

                sim_screen.draw_pnts(sim_mpcstl.traj_l_d_rec[:, 0:2], 2, sim_mpcstl.hcolor_lane_d)
                sim_screen.draw_pnts(sim_mpcstl.traj_l_u_rec[:, 0:2], 2, sim_mpcstl.hcolor_lane_u)

                for nidx_seg in range(0, len(sim_track.pnts_m_track)):
                    pnts_m_track = sim_track.pnts_m_track[nidx_seg]
                    for nidx_lane in range(0, len(pnts_m_track)):
                        pnts_m_lane = pnts_m_track[nidx_lane]
                        sim_screen.draw_pnts(pnts_m_lane[:, 0:2], 1.5, get_rgb("Deep Pink"))

                # Draw goal
                sim_screen.draw_pnts(state_array_goal[:, 0:2], 3, get_rgb("Orange"), is_fill=True)
                sim_screen.draw_pnt(state_goal, 5, get_rgb('Red'), is_fill=False)

                # Draw text
                sim_screen.draw_texts_nearby(data_ev_t, data_ov_t, id_near, 20, get_rgb("Black"), is_bold=0,
                                             sub_num=0)

            if MODE_SCREEN == 1:  # DRAW-SUB
                sim_screen.draw_basic_sub()
                sim_screen.draw_pnts(data_ov_t[:, 1:3], 2.0, get_rgb("Salmon"), sub_num=1)
                sim_screen.draw_pnt(s_ev_t[0:2], 2.5, get_rgb("Cyan"), sub_num=1)

            if PLOT_DEBUG > 0:
                sim_screen.draw_basic_sub_mpcstl([0, 0], sim_mpcstl, x_out, rmin_l_d=rmin_l_r, rmin_l_u=rmin_l_l)

            sim_screen.display()  # DISPLAY

            # SAVE PIC
            if SAVE_RESULT == 2:
                filename2save_pic = "{:s}/{:d}_{:d}.png".format(directory2save_pic, PLOT_DEBUG, t_cur - t_init)
                sim_screen.save_image(filename2save_pic)

    print("[SIMULATION END]--------------------------------------------------")

    is_break = is_outside or is_collision or is_reachgoal
    len_runtime = (t_cur - t_init) if is_break else (t_cur - t_init + 1)
    print("{:d}".format(len_runtime), end='')
    if is_outside:
        flag_exp = 1
        print(", OUTSIDE TRACK!")
    elif is_collision:
        flag_exp = 2
        print(", COLLISION!")
    elif is_reachgoal:
        flag_exp = 3
        print(", REACH GOAL!")
    else:
        flag_exp = 0
        print("")

    # Update repository
    idx_update_rep = np.arange(0, len_runtime)
    r_measured_hist = r_measured_hist[idx_update_rep, :]
    cost_measured_hist = cost_measured_hist[idx_update_rep, :]
    error_hist = error_hist[idx_update_rep, :]

    # Error summary
    summary_trial_cnt = np.zeros(sim_mpcstl.maxnum_trial, dtype=np.int32)
    summary_trial_pct = np.zeros(sim_mpcstl.maxnum_trial, dtype=np.float32)
    for nidx_trial in range(0, sim_mpcstl.maxnum_trial):
        idx_found_trial = np.where(error_hist[:, 0] == (nidx_trial + 1))
        idx_found_trial = idx_found_trial[0]
        summary_trial_cnt[nidx_trial] = int(len(idx_found_trial))
        summary_trial_pct[nidx_trial] = len(idx_found_trial) / len_runtime * 100

    idx_found_error = np.where(error_hist[:, 1] == 1)
    idx_found_error = idx_found_error[0]
    summary_error_cnt, summary_error_pct = int(len(idx_found_error)), len(idx_found_error) / len_runtime * 100

    idx_found_error_until = np.where(error_hist[:, 2] == 1)
    idx_found_error_until = idx_found_error_until[0]
    summary_error_until_cnt = int(len(idx_found_error_until))
    summary_error_until_pct = len(idx_found_error_until) / len_runtime * 100

    print("[SUMMARY]--------------------------------------------------")
    print("Trackname: {:s}, Trackid: {:s}, Vehicle-id: {:d}, Length: {:d}".format(trackname_track, trackname_id, id_ev,
                                                                                  len_runtime))
    print("Trial: (1, {:d}, {:.2f}%), (2, {:d}, {:.2f}%), (3, {:d}, {:.2f}%)".
          format(summary_trial_cnt[0], summary_trial_pct[0], summary_trial_cnt[1], summary_trial_pct[1],
                 summary_trial_cnt[2], summary_trial_pct[2]))
    print("Error: {:d}, {:.2f}%, Error-until: {:d}, {:.2f}%".format(summary_error_cnt, summary_error_pct,
                                                                    summary_error_until_cnt, summary_error_until_pct))

    # Save results
    if SAVE_RESULT >= 1:
        r_init = np.array([rmin_l_r, rmin_l_l, rmin_obs_cf, rmin_obs_o, rmin_vel, rmin_until], dtype=np.float32)
        result2save = {'trackname_track': trackname_track, 'trackname_id': trackname_id, 'id_ev': id_ev,
                       'len_runtime': len_runtime,
                       'summary_trial_cnt': summary_trial_cnt, 'summary_trial_pct': summary_trial_pct,
                       'summary_error_cnt': summary_error_cnt, 'summary_error_pct': summary_error_pct,
                       'summary_error_until_cnt': summary_error_until_cnt, 'summary_error_until_pct': summary_error_until_pct,
                       'data_ev_t_exp': data_ev_t_exp, 'data_ov_t_exp': data_ov_t_exp, 's_ev_t_exp': s_ev_t_exp,
                       'id_near_t_exp': id_near_t_exp, 'id_rest_t_exp': id_rest_t_exp,
                       'traj_ov_near_exp': traj_ov_near_exp, 'size_ov_near_exp': size_ov_near_exp,
                       'y_ev_exp': y_ev_exp, 'traj_ev_exp': traj_ev_exp, 'flag_exp': flag_exp,
                       'r_init': r_init, 'r_measured_hist': r_measured_hist,
                       'cost_measured_hist': cost_measured_hist, 'error_hist': error_hist}
        filename2save_result = directory2save + '/result'
        np.save(filename2save_result, result2save)
        print("[SAVED]")
    print("[DONE]--------------------------------------------------")


if __name__ == "__main__":
    # Parse arguments and use defaults when needed
    parser = argparse.ArgumentParser(description='Main script for testing mpcstl')
    parser.add_argument('--MODE_INIT', type=int, default=1, help='0: Add new vehicle // '
                                                                 '1: Remove existing vehicle and replace with new one')
    parser.add_argument('--VIEW_SCREEN', type=int, default=1, help='Whether to view screen or not')
    parser.add_argument('--MODE_SCREEN', type=int, default=1, help='0: plot all track // 1: plot part of track')
    parser.add_argument('--PLOT_BK', type=int, default=0, help='0: White // 1: Black')
    parser.add_argument('--PLOT_DEBUG', type=int, default=2, help='Plot for debugging 0 ~ 2 '
                                                                  '(1: +draw second-sub, 2: +draw additional-info)')
    parser.add_argument('--SAVE_RESULT', type=int, default=0, help='0: none, 1: +save result, 2: +save pic')
    parser.add_argument('--VERBOSE', type=int, default=2, help='Print verbose (0: simple, '
                                                               '1: +check the success of optimization, +robustness prediction, '
                                                               '2: +detailed optimization result)')

    parser.add_argument('--trackname', type=str, default='us101_2', help='Track name: us101_1 ~ us101_3, '
                                                                          'i80_1 ~ i80_3, highd_1 ~ highd_60')
    parser.add_argument('--v_id', type=int, default=1221, help='vehicle-id')
    args = parser.parse_args()

    key_params = {'MODE_INIT': args.MODE_INIT, 'VIEW_SCREEN': args.VIEW_SCREEN, 'MODE_SCREEN': args.VIEW_SCREEN,
                  'PLOT_BK': args.PLOT_BK, 'PLOT_DEBUG': args.PLOT_DEBUG, 'SAVE_RESULT': args.SAVE_RESULT,
                  'VERBOSE': args.VERBOSE, 'trackname': args.trackname, 'v_id': args.v_id}

    main(key_params)
