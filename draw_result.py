# DRAW CONTROL-RESULT
#   : Screen with matplotlib

from __future__ import print_function

import os
import argparse
import time

from core.SimScreenMatplotlib import SimScreenMatplotlib
from core.SimTrack import *
from src.utils import *
from src.utils_sim import *
from src.utils_stl import *
from src.get_rgb import *

import matplotlib
import matplotlib.pyplot as plt


def main(params):
    # SET KEY PARAMETERS ----------------------------------------------------------------------------------------------#
    MODE_SCREEN, PLOT_BK, PLOT_R = params['MODE_SCREEN'], params['PLOT_BK'], params['PLOT_R']

    # Test environment parameter
    trackname, id_ev = params['trackname'], params['v_id']

    # Model-name to load
    model_name = params['model_name']

    # Set directory to load
    if id_ev < 0:
        directory2read = "./result/{:s}/{:s}_idneg".format(model_name, trackname)
    else:
        directory2read = "./result/{:s}/{:s}_id{:d}".format(model_name, trackname, id_ev)

    # LOAD RESULT -----------------------------------------------------------------------------------------------------#
    r_filename = directory2read + '/result.npy'
    data_read_tmp = np.load(r_filename, allow_pickle=True)
    data_ev_t_exp = data_read_tmp[()]['data_ev_t_exp']
    data_ov_t_exp = data_read_tmp[()]['data_ov_t_exp']
    s_ev_t_exp = data_read_tmp[()]['s_ev_t_exp']
    id_near_t_exp = data_read_tmp[()]['id_near_t_exp']
    id_rest_t_exp = data_read_tmp[()]['id_rest_t_exp']
    traj_ov_near_exp = data_read_tmp[()]['traj_ov_near_exp']
    size_ov_near_exp = data_read_tmp[()]['size_ov_near_exp']
    y_ev_exp = data_read_tmp[()]['y_ev_exp']
    traj_ev_exp = data_read_tmp[()]['traj_ev_exp']
    flag_exp = data_read_tmp[()]['flag_exp']
    if PLOT_R == 1:
        r_pred_hist = data_read_tmp[()]['r_pred_hist']
        r_pred_sigma_hist = data_read_tmp[()]['r_pred_sigma_hist']
        r_measured_hist = data_read_tmp[()]['r_measured_hist']
    len_runtime = len(data_ev_t_exp)

    # SET DIRECTORY PIC -----------------------------------------------------------------------------------------------#
    directory2save_pic = "{:s}/pic_draw".format(directory2read)
    if not os.path.exists(directory2save_pic):
        os.makedirs(directory2save_pic)

    if PLOT_R == 1:
        idx_r_plot = [0, 1, 2, 4, 5]
        print("[DRAW START]--------------------------------------------------")
        rmin_hist = r_pred_hist - r_pred_sigma_hist

        rmin_max = np.amax(rmin_hist, axis=0)
        rmin_min = np.amin(rmin_hist, axis=0)
        rmin_max_ = np.abs(np.reshape(rmin_max, (1, -1)))
        rmin_min_ = np.abs(np.reshape(rmin_min, (1, -1)))
        rmin_absmax_ = np.concatenate((rmin_max_,rmin_min_), axis=0)
        rmin_absmax = np.amax(rmin_absmax_, axis=0)

        for idx_t in range(0, len_runtime):
            print("{:d}/{:d}".format(idx_t, len_runtime))
            r_measured_sel = r_measured_hist[idx_t, :]
            rmin_sel = rmin_hist[idx_t, :]

            r_measured_plot = r_measured_sel[idx_r_plot] / rmin_absmax[idx_r_plot]
            rmin_plot = rmin_sel[idx_r_plot] / rmin_absmax[idx_r_plot]

            r_measured_plot = set_vector_in_range(r_measured_plot, -1, +1)
            rmin_plot = set_vector_in_range(rmin_plot, -1, +1)

            filename2save_r0 = "{:s}/r0_b{:d}_{:d}.png".format(directory2save_pic, PLOT_BK, idx_t)
            plot_robustness_bar_twin(r_measured_plot, rmin_plot, 0.3, 1, filename2save_r0, get_rgb('Pastel Red'),
                                     get_rgb('Pastel Blue'), is_background_black=PLOT_BK)

            filename2save_r1 = "{:s}/r1_b{:d}_{:d}.png".format(directory2save_pic, PLOT_BK, idx_t)
            plot_robustness_bar(rmin_plot, 0.6, 1, filename2save_r1, get_rgb('Pastel Red'), get_rgb('Pastel Blue'),
                                is_background_black=PLOT_BK)

    else:
        # SET TRACK ---------------------------------------------------------------------------------------------------#
        sim_track = SimTrack(trackname)

        # SET SCREEN --------------------------------------------------------------------------------------------------#
        if MODE_SCREEN == 1:
            height2width_ratio = 1.8 / 3.0
            screen_width = 7.5
            screen_xlen = 85
            dpi_save = 200
        else:
            track_xlen = sim_track.pnt_max[0] - sim_track.pnt_min[0] + 75.0
            track_ylen = sim_track.pnt_max[1] - sim_track.pnt_min[1] + 15.0
            track_xmean = (sim_track.pnt_max[0] + sim_track.pnt_min[0]) * 0.5
            track_ymean = (sim_track.pnt_max[1] + sim_track.pnt_min[1]) * 0.5
            height2width_ratio = track_ylen / track_xlen
            screen_width = 16.0
            dpi_save = 400

        sim_screen_m = SimScreenMatplotlib(MODE_SCREEN, PLOT_BK)
        sim_screen_m.set_figure(screen_width, screen_width * height2width_ratio)
        sim_screen_m.set_pnts_track_init(sim_track.pnts_poly_track, sim_track.pnts_outer_border_track,
                                         sim_track.pnts_inner_border_track)

        # DRAW --------------------------------------------------------------------------------------------------------#
        print("[DRAW START]--------------------------------------------------")
        for idx_t in range(0, len_runtime):
            print("{:d}/{:d}".format(idx_t, len_runtime))
            data_ev_t, data_ov_t = data_ev_t_exp[idx_t], data_ov_t_exp[idx_t]
            s_ev_t = s_ev_t_exp[idx_t]
            id_near_t, id_rest_t = id_near_t_exp[idx_t], id_rest_t_exp[idx_t]
            traj_ovnear_t, size_ovnear_t = traj_ov_near_exp[idx_t], size_ov_near_exp[idx_t]
            y_ev_t = y_ev_exp[idx_t]
            traj_hist_ev = traj_ev_exp[idx_t]

            if idx_t == 0:
                data_ev_t0, data_ov_t0 = np.copy(data_ev_t), np.copy(data_ov_t)

            sim_screen_m.set_figure(screen_width, screen_width * height2width_ratio)
            if MODE_SCREEN == 1:
                sim_screen_m.set_pnt_range([s_ev_t[0], s_ev_t[1]], [screen_xlen, screen_xlen * height2width_ratio])
            else:
                sim_screen_m.set_pnt_range([track_xmean, track_ymean], [track_xlen, track_ylen])

            # Draw track
            sim_screen_m.draw_track()

            if MODE_SCREEN == 1:
                sim_screen_m.draw_ctrl_basic(data_ev_t, data_ov_t, id_near_t, id_rest_t, y_ev_t, traj_hist_ev,
                                             traj_ovnear_t, size_ovnear_t, v_arrow=20)
            else:
                sim_screen_m.draw_basic(data_ev_t, data_ov_t, id_near_t, id_rest_t, traj_hist_ev)

            sim_screen_m.update_view_range()
            plt.show(block=False)
            time.sleep(0.5)

            # SAVE-PIC
            plt.axis('off')
            filename2save_r0 = "{:s}/m{:d}b{:d}_{:d}.png".format(directory2save_pic, MODE_SCREEN, PLOT_BK, idx_t)
            sim_screen_m.save_figure(filename2save_r0, dpi=dpi_save)
            time.sleep(0.5)
            plt.close('all')

    print("[DRAW END]--------------------------------------------------")
    print("[DONE]--------------------------------------------------")


if __name__ == "__main__":
    # Parse arguments and use defaults when needed
    parser = argparse.ArgumentParser(description='Main script for drawing result.')
    parser.add_argument('--MODE_SCREEN', type=int, default=1, help='0: plot all track // 1: plot part of track')
    parser.add_argument('--PLOT_BK', type=int, default=0, help='0: White // 1: Black')
    parser.add_argument('--PLOT_R', type=int, default=0, help='Whether to plot robustness')
    parser.add_argument('--model_name', type=str, default='lbmpcstl_gpr', help='model name')
    parser.add_argument('--trackname', type=str, default='us101_1', help='Track name: us101_1 ~ us101_3, '
                                                                         'i80_1 ~ i80_3, highd_1 ~ highd_60')
    parser.add_argument('--v_id', type=int, default=141, help='vehicle-id')
    args = parser.parse_args()

    key_params = {'MODE_SCREEN': args.MODE_SCREEN, 'PLOT_BK': args.PLOT_BK, 'PLOT_R': args.PLOT_R,
                  'model_name': args.model_name, 'trackname': args.trackname, 'v_id': args.v_id}

    main(key_params)
