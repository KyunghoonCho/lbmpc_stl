# SET SCREEN FOR SIMULATOR (MATPLOTLIB)
#   - Matplotlib-based simulator

from __future__ import print_function


import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from src.utils import *
from src.get_rgb import *
from matplotlib.patches import Polygon


class SimScreenMatplotlib(object):
    """
    SCREEN FOR SIMULATOR (MATPLOTLIB)
    z-order summary:
        0: track-polygon
        1: track-lines
        2: vehicle-fill
        3: (density or cost-map)
        4: trajectory, point
        5: vehicle-border
        6: arrow
    """
    def __init__(self, mode_screen=1, plot_bk=0):
        # Points of range (screen)
        self.pnt_mean, self.pnt_range = [], []
        self.pnt_xlim, self.pnt_ylim = [], []

        # Points (track)
        self.pnts_poly_track, self.pnts_outer_border_track, self.pnts_inner_border_track = [], [], []

        self.fig, self.axs = [], []
        self.mode_screen, self.plot_bk = mode_screen, plot_bk

    # SET PARAMETERS --------------------------------------------------------------------------------------------------#
    def set_pnt_range(self, pnt_mean, pnt_range):
        """
        Sets view-range.
        """
        self.pnt_mean = pnt_mean
        self.pnt_range = pnt_range

        self.pnt_xlim = np.array([self.pnt_mean[0] - self.pnt_range[0] / 2, self.pnt_mean[0] + self.pnt_range[0] / 2],
                                 dtype=np.float32)
        self.pnt_ylim = np.array([self.pnt_mean[1] - self.pnt_range[1] / 2, self.pnt_mean[1] + self.pnt_range[1] / 2],
                                 dtype=np.float32)

    def set_pnts_track_init(self, pnts_poly_track, pnts_outer_border_track, pnts_inner_border_track):
        """
        Sets track-points.
        """
        self.pnts_poly_track = pnts_poly_track
        self.pnts_outer_border_track = pnts_outer_border_track
        self.pnts_inner_border_track = pnts_inner_border_track

    # SET FIGURE ------------------------------------------------------------------------------------------------------#
    def set_figure(self, fig_w_size, fig_h_size, grid=False):
        """
        Sets figure.
        """
        _fig, _axs = plt.subplots(1, 1, tight_layout=True)
        _fig.set_size_inches(fig_w_size, fig_h_size)

        _axs.axis("equal")

        if grid:
            plt.grid(True)

        self.fig, self.axs = _fig, _axs

        if self.plot_bk == 1:
            self.axs.set_facecolor("black")

    def update_view_range(self):
        """
        Updates view-range.
        """
        plt.xlim(self.pnt_xlim)
        plt.ylim(self.pnt_ylim)

    # SAVE FIGURE -----------------------------------------------------------------------------------------------------#
    def save_figure(self, filename2save, dpi):
        """
        Saves figure.
        """
        self.fig.savefig(filename2save, dpi=dpi)

    # DRAW (MAIN) -----------------------------------------------------------------------------------------------------#
    def draw_track(self, zorder=1):
        """
        Draws track.
        """
        # pnts_poly_track: (list) points of track

        if self.mode_screen == 1:
            linewidth_outer, linewidth_inner = 1.0, 0.7
        else:
            linewidth_outer, linewidth_inner = 0.6, 0.4

        if self.plot_bk == 1:
            hcolor_track = get_rgb("Dark Gray")
            hcolor_outer_line = get_rgb("White")
            hcolor_inner_line = get_rgb("White Smoke")
        else:
            hcolor_track = get_rgb("Gray")
            hcolor_outer_line = get_rgb("Very Dark Gray")
            hcolor_inner_line = get_rgb("White Smoke")

        # Plot polygon
        zorder_polygon = max(0, zorder - 1)
        for nidx_seg in range(0, len(self.pnts_poly_track)):
            pnts_poly_seg = self.pnts_poly_track[nidx_seg]

            # Plot lane-segment
            for nidx_lane in range(0, len(pnts_poly_seg)):
                idx_lane = len(pnts_poly_seg) - nidx_lane - 1
                # Pnts on lane-segment
                pixel_poly_lane = pnts_poly_seg[idx_lane]
                pixel_poly_lane_added = pixel_poly_lane
                polygon_track = Polygon(pixel_poly_lane_added, facecolor=hcolor_track, zorder=zorder_polygon)
                self.axs.add_patch(polygon_track)

        # Plot track (outer)
        for nidx_seg in range(0, len(self.pnts_outer_border_track)):
            pnts_outer_seg = self.pnts_outer_border_track[nidx_seg]
            for nidx_lane in range(0, len(pnts_outer_seg)):
                pnts_outer_seg_sel = pnts_outer_seg[nidx_lane]
                self.axs.plot(pnts_outer_seg_sel[:, 0], pnts_outer_seg_sel[:, 1], linewidth=linewidth_outer,
                              color=hcolor_outer_line, zorder=zorder, solid_joinstyle='round', solid_capstyle='round')

        # Plot track (inner)
        for nidx_seg in range(0, len(self.pnts_inner_border_track)):
            pnts_inner_seg = self.pnts_inner_border_track[nidx_seg]
            for nidx_lane in range(0, len(pnts_inner_seg)):
                pnts_inner_lane = pnts_inner_seg[nidx_lane]
                self.axs.plot(pnts_inner_lane[:, 0], pnts_inner_lane[:, 1], linestyle="--", dashes=[5, 2],
                              linewidth=linewidth_inner, color=hcolor_inner_line, zorder=zorder,
                              solid_joinstyle='round', solid_capstyle='round')

        if self.plot_bk == 1:
            self.axs.set_facecolor("black")

    def draw_vehicle_fill(self, data_v, ids, hcolor, op=1.0, zorder=2):
        """
        Draws vehicle (fill).
        :param data_v: t x y theta v length width tag_segment tag_lane id (width > length) (dim = N x 10)
        :param ids: vehicle ids (int)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot order (int)
        """
        data_v = make_numpy_array(data_v, keep_1dim=False)

        num_vehicle = data_v.shape[0]
        for nidx_n in range(0, num_vehicle):
            # Get vehicle-data
            #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
            data_v_tmp = data_v[nidx_n, :]

            # Get polygon-points
            pnts_v_tmp = get_pnts_carshape(data_v_tmp[1], data_v_tmp[2], data_v_tmp[3], data_v_tmp[6], data_v_tmp[5])

            # Set (fill) color
            if len(ids) == 0:
                hcolor_sel = get_rgb("Light Gray")
            elif np.isin(data_v_tmp[-1], ids):
                hcolor_sel = hcolor
            else:
                hcolor_sel = get_rgb("Light Gray")

            # Plot vehicle
            polygon_vehicle = Polygon(pnts_v_tmp, facecolor=hcolor_sel, zorder=zorder, alpha=op)
            self.axs.add_patch(polygon_vehicle)

    def draw_vehicle_border(self, data_v, ids, hlw, hcolor, op=1.0, zorder=5):
        """
        Draws vehicle border.
        :param data_v: t x y theta v length width tag_segment tag_lane id (width > length) (dim = N x 10)
        :param ids: vehicle ids (int)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot order (int)
        """
        data_v = make_numpy_array(data_v, keep_1dim=False)
        idx_found = np.where(ids > -1)
        idx_found = idx_found[0]

        for nidx_d in range(0, idx_found.shape[0]):
            id_sel_tmp = ids[idx_found[nidx_d]]

            idx_tmp = np.where(data_v[:, -1] == id_sel_tmp)
            idx_tmp = idx_tmp[0]

            data_v_tmp = data_v[idx_tmp, :]
            data_v_tmp = make_numpy_array(data_v_tmp, keep_1dim=True)

            # Get polygon-points
            pnts_v_tmp_ = get_pnts_carshape(data_v_tmp[1], data_v_tmp[2], data_v_tmp[3], data_v_tmp[6], data_v_tmp[5])
            pnts_v_tmp_0 = pnts_v_tmp_[0, :]
            pnts_v_tmp_0 = np.reshape(pnts_v_tmp_0, (1, -1))
            pnts_v_tmp = np.concatenate((pnts_v_tmp_, pnts_v_tmp_0), axis=0)

            # Plot vehicle
            self.axs.plot(pnts_v_tmp[:, 0], pnts_v_tmp[:, 1], linewidth=hlw, color=hcolor, zorder=zorder,
                          solid_joinstyle='round', solid_capstyle='round', alpha=op)

    def draw_target_vehicle_fill(self, x, y, theta, rx, ry, hcolor, op=1.0, zorder=2):
        """
        Draws target-vehicle (fill).
        :param x: position x (float)
        :param y: position y (float)
        :param theta: heading (rad) (float)
        :param rx: vehicle length (float)
        :param ry: vehicle width (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot order (int)
        """
        # Get polygon-points
        pnts_v_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        # Plot vehicle
        polygon_vehicle = Polygon(pnts_v_tmp, facecolor=hcolor, zorder=zorder, alpha=op)
        self.axs.add_patch(polygon_vehicle)

    def draw_target_vehicle_border(self, x, y, theta, rx, ry, hlw, hls='-', hcolor='k', op=1.0, zorder=5):
        """
        Draws target vehicle border.
        :param x: position x (float)
        :param y: position y (float)
        :param theta: heading (rad) (float)
        :param rx: vehicle length (float)
        :param ry: vehicle width (float)
        :param hlw: linewidth (float)
        :param hls: linestyle (string)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot order (int)
        """
        # Get polygon-points
        pnts_v_tmp_ = get_pnts_carshape(x, y, theta, rx, ry)
        pnts_v_tmp_0 = pnts_v_tmp_[0, :]
        pnts_v_tmp_0 = np.reshape(pnts_v_tmp_0, (1, -1))
        pnts_v_tmp = np.concatenate((pnts_v_tmp_, pnts_v_tmp_0), axis=0)

        # Plot vehicle
        self.axs.plot(pnts_v_tmp[:, 0], pnts_v_tmp[:, 1], linewidth=hlw, linestyle=hls, color=hcolor, zorder=zorder,
                      solid_joinstyle='round', solid_capstyle='round', alpha=op)

    def draw_pnts_scatter(self, pnts, s, c, op=1.0, zorder=4):
        """
        Draws points (scatter).
        """
        self.axs.scatter(pnts[:, 0], pnts[:, 1], s=s, c=c, alpha=op, zorder=zorder)

    def draw_traj(self, traj, hlw, hls, hcolor, op=1.0, zorder=4):
        """
        Draws trajectory.
        :param traj: trajectory (dim = N x 2)
        :param hlw: linewidth (float)
        :param hls: linestyle (string)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        # Plot
        if op == -1:
            self.axs.plot(traj[:, 0], traj[:, 1], linewidth=hlw, linestyle=hls, color=hcolor, zorder=zorder,
                          solid_joinstyle='round', solid_capstyle='round')
        else:
            self.axs.plot(traj[:, 0], traj[:, 1], linewidth=hlw, linestyle=hls, color=hcolor, alpha=op, zorder=zorder,
                          solid_joinstyle='round', solid_capstyle='round')

    def draw_trajs(self, trajs, hlw, hls, hcolor, op=1.0, zorder=4):
        """
        Draws trajectories.
        :param trajs: trajectories (list)
        :param hlw: linewidth (float)
        :param hls: linestyle (string)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        if len(trajs) > 0:
            num_traj = len(trajs)

            for nidx_traj in range(0, num_traj):  # Plot trajectories
                traj_sel = trajs[nidx_traj]
                self.draw_traj(traj_sel[:, 0:2], hlw, hls, hcolor, op, zorder)

    def draw_trajs_w_cost(self, trajs, costs, idx_invalid, cost_max, hlw, hls, hcolor_map="cool", op=1.0, zorder=4):
        """
        Draws trajectories w.r.t. cost.
        :param trajs: trajectories (list)
        :param costs: costs of trajectories (dim = N)
        :param idx_invalid: invalid trajectory indexes (list)
        :param cost_max: maximum-cost (if 0, cost is scaled automatically)
        :param hlw: linewidth (float)
        :param hls: linestyle (string)
        :param hcolor_map: matplotlib-colormap name (string)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        if len(trajs) > 0:
            num_traj = len(trajs)
            if len(costs) > 0:
                costs = make_numpy_array(costs, keep_1dim=True)
            else:
                costs = np.array((num_traj,), dtype=np.float32)

            if len(idx_invalid) > 0:
                idx_invalid = make_numpy_array(idx_invalid, keep_1dim=True)
            else:
                idx_invalid = np.array([])

            if len(idx_invalid) > 0:
                idx_valid = np.setdiff1d(np.arange(0, num_traj), idx_invalid)
            else:
                idx_valid = np.arange(0, num_traj)

            if len(idx_valid) > 0:
                cost_valid = costs[idx_valid]
                if cost_max == 0:
                    # mean_cost_valid, std_cost_valid = np.mean(cost_valid), np.std(cost_valid)
                    mean_cost_valid, std_cost_valid = 0.0, np.std(cost_valid)
                    min_cost_valid = mean_cost_valid - 2 * std_cost_valid
                    max_cost_valid = mean_cost_valid + 2 * std_cost_valid
                    if abs(max_cost_valid - min_cost_valid) < float(1e-4):
                        max_cost_valid, min_cost_valid = 1.0, 0.0
                else:
                    max_cost_valid, min_cost_valid = cost_max, 0.0
            else:
                max_cost_valid, min_cost_valid = 1.0, 0.0

            for nidx_traj in range(0, idx_invalid.shape[0]):  # Plot invalid trajectories
                idx_sel = idx_invalid[nidx_traj]
                traj_sel = trajs[idx_sel]
                self.draw_traj(traj_sel[:, 0:2], hlw, hls, get_rgb("Dark Slate Blue"), op, zorder)

            cmap = matplotlib.cm.get_cmap(hcolor_map)
            for nidx_traj in range(0, idx_valid.shape[0]):  # Plot valid trajectories
                idx_sel = idx_valid[nidx_traj]
                traj_sel = trajs[idx_sel]

                idx_tmp = (costs[idx_sel] - min_cost_valid) / (max_cost_valid - min_cost_valid)
                idx_tmp = min(max(idx_tmp, 0.0), 1.0)
                cmap_tmp = cmap(idx_tmp)
                self.draw_traj(traj_sel[:, 0:2], hlw, hls, cmap_tmp, op, zorder)

    def draw_vehicle_border_traj(self, traj, rx, ry, stepsize, hlw, hcolor, op=1.0, zorder=4):
        """
        Draws vehicle border trajectory.
        :param traj: trajectory (dim = N x 3)
        :param rx: vehicle length (float)
        :param ry: vehicle width (float)
        :param stepsize: step-size to plot (int)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        for nidx_t in range(stepsize, traj.shape[0], stepsize):
            pnt_sel = traj[nidx_t, :]

            if ~np.isnan(pnt_sel[0]):
                self.draw_target_vehicle_border(pnt_sel[0], pnt_sel[1], pnt_sel[2], rx, ry, hlw, hcolor=hcolor, op=op,
                                                zorder=zorder)

    def draw_vehicle_border_trajs(self, trajs, sizes, stepsize, hlinewidth, hcolor, op=1.0, zorder=4):
        """ Draws vehicle border trajectories.
        :param trajs: list of trajectories (ndarray, dim = N x 3)
        :param sizes: list of sizes rx, ry (ndarray, dim = N x 2)
        :param stepsize: step-size (int)
        :param hlinewidth: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        for nidx_traj in range(0, len(trajs)):
            traj_sel = trajs[nidx_traj]
            size_sel = sizes[nidx_traj, :]
            rx_sel = size_sel[0]
            ry_sel = size_sel[1]

            self.draw_vehicle_border_traj(traj_sel, rx_sel, ry_sel, stepsize, hlinewidth, hcolor, op=op, zorder=zorder)

    def draw_vehicle_fill_traj_cgad(self, traj, rx, ry, stepsize, hcolor1, hcolor2, op=1.0, zorder=2):
        """ Draws vehicle fill trajectory with color gradient.
        :param traj: trajectory (dim = N x 3)
        :param rx: vehicle size (x) (float)
        :param ry: vehicle size (y) (float)
        :param stepsize: step-size (int)
        :param hcolor1: start rgb color (tuple)
        :param hcolor2: end rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        len_traj = traj.shape[0]
        len_color = math.floor((len_traj - 1) / (stepsize))
        cnt_color = 0
        for nidx_t in range(stepsize, len_traj, stepsize):
            pnt_sel = traj[nidx_t, :]  # Point to draw

            # Set new color
            cnt_color += 1
            alpha = float(cnt_color / (len_color + 1.0))
            hcolor_sel = get_color_mix(alpha, hcolor1, hcolor2)

            if ~np.isnan(pnt_sel[0]):
                self.draw_target_vehicle_fill(pnt_sel[0], pnt_sel[1], pnt_sel[2], rx, ry, hcolor_sel, op=op, zorder=zorder)

    def draw_vehicle_fill_trajs_cgad(self, trajs, sizes, stepsize, hcolor1, hcolor2, op=1.0, zorder=2):
        """ Draws vehicle fill trajectories with color gradient.
        :param trajs: list of trajectories (ndarray, dim = N x 3)
        :param sizes: list of sizes rx, ry (ndarray, dim = N x 2)
        :param stepsize: step-size (int)
        :param hlinewidth: linewidth (float)
        :param hcolor1: start rgb color (tuple)
        :param hcolor2: end rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        for nidx_traj in range(0, len(trajs)):
            traj_sel, size_sel = trajs[nidx_traj], sizes[nidx_traj, :]
            rx_sel, ry_sel = size_sel[0], size_sel[1]
            self.draw_vehicle_fill_traj_cgad(traj_sel, rx_sel, ry_sel, stepsize, hcolor1, hcolor2, op=op, zorder=zorder)

    def draw_vehicle_arrow(self, data_v, ids, lv_max, hcolor, op=1.0, zorder=6):
        """
        Draws vehicle arrow.
        :param data_v: t x y theta v length width tag_segment tag_lane id (width > length) (dim = N x 10)
        :param ids: vehicle ids (list)
        :param lv_max: maximum linear-velocity (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        if len(data_v) > 0:
            data_v = make_numpy_array(data_v, keep_1dim=False)

            idx_found = np.where(ids > -1)
            idx_found = idx_found[0]

            for nidx_d in range(0, idx_found.shape[0]):
                id_near_sel = ids[idx_found[nidx_d]]

                idx_tmp = np.where(data_v[:, -1] == id_near_sel)
                idx_tmp = idx_tmp[0]

                if len(idx_tmp) > 0:
                    data_v_sel = data_v[idx_tmp, :]
                    data_v_sel = data_v_sel.reshape(-1)
                    self.draw_target_vehicle_arrow(data_v_sel[1], data_v_sel[2], data_v_sel[3], data_v_sel[6],
                                                   data_v_sel[5], data_v_sel[4], lv_max, hcolor, op=op, zorder=zorder)

    def draw_target_vehicle_arrow(self, x, y, theta, rx, ry, lv, lv_max, hcolor, op=1.0, zorder=6):
        """
        Draws target vehicle arrow.
        :param x: position x (float)
        :param y: position y (float)
        :param theta: heading (rad) (float)
        :param rx: vehicle length (float)
        :param ry: vehicle width (float)
        :param lv: linear-velocity (float)
        :param lv_max: maximum linear-velocity (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        ratio_lv = 0.15 + (abs(lv) / lv_max) * 0.85
        ratio_lv = min(ratio_lv, 1.0)

        ax, ay = ratio_lv * rx * 0.4, ry * 0.15
        bx, by = ratio_lv * rx * 0.15, ry * 0.15

        # Get polygon-points
        pnts_v_tmp = get_pnts_arrow(x, y, theta, ax, ay, bx, by)

        # Plot arrow
        polygon_arrow = Polygon(pnts_v_tmp, facecolor=hcolor, alpha=op, zorder=zorder)
        self.axs.add_patch(polygon_arrow)

    def draw_2d_density(self, pnts, res, cover=1, sigma=1.0, beta=0.0, hcolor_map="Blues", op=1.0, zorder=3):
        """
        Draws 2D density.
        :param pnts: points (dim = N x 2)
        :param res: resolution (float)
        :param cover: cover option (0: small, 1: medium, 2: all)
        :param sigma: sigma of gaussian kde (float)
        :param beta: threshold parameter 0 ~ 1 (float)
        :param hcolor_map: matplotlib-colormap name (string)
        :param op: opacity 0 ~ 1 (float)
        :param zorder: plot-order (int)
        """
        pnts = make_numpy_array(pnts, keep_1dim=False)
        pnts = pnts[:, 0:2]

        if cover == 0:
            xmin_, xmax_ = np.amin(pnts[:, 0]), np.amax(pnts[:, 0])
            ymin_, ymax_ = np.amin(pnts[:, 1]), np.amax(pnts[:, 1])

            xmin, xmax = xmin_ - (xmax_ - xmin_) * 0.25, xmax_ + (xmax_ - xmin_) * 0.25
            ymin, ymax = ymin_ - (ymax_ - ymin_) * 0.25, ymax_ + (ymax_ - ymin_) * 0.25
        elif cover == 1:
            xmin_, xmax_ = np.amin(pnts[:, 0]), np.amax(pnts[:, 0])
            ymin_, ymax_ = np.amin(pnts[:, 1]), np.amax(pnts[:, 1])

            xrange_, yrange_ = max(xmax_ - xmin_, 5), max(ymax_ - ymin_, 5)

            xmin, xmax = xmin_ - xrange_ * 0.5, xmax_ + xrange_ * 0.5
            ymin, ymax = ymin_ - yrange_ * 0.5, ymax_ + yrange_ * 0.5
        else:
            xmin, xmax = self.pnt_xlim[0], self.pnt_xlim[1]
            ymin, ymax = self.pnt_ylim[0], self.pnt_ylim[1]

        xrange, yrange = (xmax - xmin), (ymax - ymin)
        num_x, num_y = int(xrange / res), int(yrange / res)
        num_x, num_y = max(num_x, 5), max(num_y, 5)
        x_grid_, y_grid_ = np.linspace(xmin, xmax, num_x), np.linspace(ymin, ymax, num_y)
        x_grid, y_grid = np.meshgrid(x_grid_, y_grid_)

        # Perform the kernel density estimate
        pnts_train = np.vstack([pnts[:, 0], pnts[:, 1]])
        pnts_test = np.vstack([x_grid.ravel(), y_grid.ravel()])

        if sigma < 0:
            func_kde = st.gaussian_kde(pnts_train)
            z_grid = np.transpose(np.array(func_kde(pnts_test)))
        else:
            z_grid = apply_gaussian_kde_2d_naive(np.transpose(pnts_test), np.transpose(pnts_train), sigma)
        z_grid = np.reshape(z_grid, x_grid.shape)
        z_grid = z_grid - np.amin(z_grid.reshape(-1))
        zmin_grid, zmax_grid = np.amin(z_grid.reshape(-1)), np.amax(z_grid.reshape(-1))

        z_grid_lb = zmin_grid + beta * (zmax_grid - zmin_grid)

        # idx_mask = np.where(z_grid <= z_grid_lb)
        # z_mask = np.zeros_like(z_grid, dtype=bool)
        # z_mask[idx_mask[0], idx_mask[1]] = True
        # z_map_plot = np.ma.array(z_grid, mask=z_mask)

        # Contourf plot
        # h_cntourf = self.axs.contourf(x_grid, y_grid, z_grid, levels=np.linspace(z_grid_lb, zmax_grid), cmap=hcolor_map,
        #                               alpha=op, zorder=zorder)
        # h_cntourf = self.axs.contourf(x_grid, y_grid, z_map_plot, linewidth=0, cmap="Blues", alpha=tr, zorder=10)

        x_map_r, y_map_r, z_map_r = x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)
        h_cntourf = plt.tricontourf(x_map_r, y_map_r, z_map_r, levels=np.linspace(z_grid_lb, zmax_grid),
                                    cmap=hcolor_map, alpha=op, zorder=zorder)

        # This is the fix for the white lines between contour levels
        for c in h_cntourf.collections:
            c.set_edgecolor("face")
            # c.set_linewidth(0.000001)
            c.set_linewidth(0.1)

        return x_grid, y_grid, z_grid

    # -----------------------------------------------------------------------------------------------------------------#
    # DRAW-CONTROL ----------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    def draw_basic(self, data_tv_cur, data_ov_cur, id_near, id_rest, traj_tv_hist):
        """
        Draws prediction (basic).
        """
        # Color setting
        hcolor_fill_tv = get_rgb("Crayola")
        hcolor_fill_ovsel = get_rgb("Salmon")
        hcolor_border_tv = get_rgb("Han Blue")
        hcolor_border_ovsel, hcolor_border_ovrest = get_rgb("Dark Pastel Red"), get_rgb("Dim Gray")

        hcolor_hist_traj_tv = get_rgb("Dark Turquoise")

        # Draw other vehicle
        self.draw_vehicle_fill(data_ov_cur, id_near, hcolor_fill_ovsel)

        # Draw target vehicle
        self.draw_target_vehicle_fill(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6], data_tv_cur[5],
                                      hcolor_fill_tv)

        # Draw traj-hist (ego-vehicle)
        self.draw_traj(traj_tv_hist[:, 0:2], 1.0, '-', hcolor_hist_traj_tv, 1, zorder=3)

        # Draw borders
        self.draw_vehicle_border(data_ov_cur, id_near, 0.5, hcolor_border_ovsel)
        self.draw_vehicle_border(data_ov_cur, id_rest, 0.5, hcolor_border_ovrest)
        self.draw_target_vehicle_border(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6], data_tv_cur[5],
                                        0.7, hcolor=hcolor_border_tv)

    def draw_pred_basic(self, data_tv_cur, data_ov_cur, id_near, id_rest, y_tv, y_tv_pred, traj_ov_near, size_ov_near,
                        traj_tv_hist, cost_tv_pred, hcolormap="viridis"):
        """
        Draws prediction (basic).
        """
        # Color setting
        hcolor_fill_tv = get_rgb("Crayola")
        hcolor_fill_ovsel, hcolor_fill_rest = get_rgb("Salmon"), get_rgb("Dark Gray")
        hcolor_arrow_sel, hcolor_arrow_rest = get_rgb("White"), get_rgb("Dark Gray")
        hcolor_border_tv = get_rgb("Han Blue")
        hcolor_border_ovsel, hcolor_border_ovrest = get_rgb("Dark Pastel Red"), get_rgb("Dim Gray")

        hcolor_traj_tv = get_rgb("Dodger Blue")  # "Han Blue"
        hcolor_traj_ov = get_rgb("Indian Red")
        hcolor_border_traj_tv = get_rgb("Corn Flower Blue")
        hcolor_hist_traj_tv = get_rgb("Dark Turquoise")

        if len(cost_tv_pred) > 0:
            cost_tv_pred_min, cost_tv_pred_max = min(cost_tv_pred), max(cost_tv_pred)
        else:
            cost_tv_pred_min, cost_tv_pred_max = 0.0, 1.0

        # Draw other vehicle
        self.draw_vehicle_fill(data_ov_cur, id_near, hcolor_fill_ovsel, op=0.9)

        # Draw target vehicle
        self.draw_target_vehicle_fill(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                      data_tv_cur[5], hcolor_fill_tv, op=0.9)

        # Draw trajectories (other vehicle)
        self.draw_trajs(traj_ov_near, 0.9, '-', hcolor_traj_ov, op=0.5)
        self.draw_vehicle_fill_trajs_cgad(traj_ov_near, size_ov_near, 2, hcolor_fill_ovsel, hcolor_fill_rest, op=0.6)
        self.draw_vehicle_border_trajs(traj_ov_near, size_ov_near, 2, 0.5, hcolor_traj_ov, op=0.75)

        # Draw trajectories (target vehicle)
        cmap = matplotlib.cm.get_cmap(hcolormap)
        for nidx_n in range(0, len(y_tv_pred)):
            y_tv_tmp = y_tv_pred[nidx_n]
            if len(cost_tv_pred) > 0:
                cmap_idx = (cost_tv_pred[nidx_n] - cost_tv_pred_min) / (cost_tv_pred_max - cost_tv_pred_min)
                cmap_sel = cmap(cmap_idx)
            else:
                cmap_sel = cmap((nidx_n - 1) / (len(y_tv_pred)))

            self.draw_traj(y_tv_tmp[:, 0:2], 2.5, '-', cmap_sel, op=0.62)
            self.draw_target_vehicle_fill(y_tv_tmp[-1, 0], y_tv_tmp[-1, 1], y_tv_tmp[-1, 2], data_tv_cur[6],
                                          data_tv_cur[5], hcolor=cmap_sel, op=0.15)

        y_tv = interpolate_traj(y_tv, alpha=5)
        self.draw_traj(y_tv[:, 0:2], 1.0, '--', hcolor_traj_tv, op=0.8, zorder=5)
        self.draw_target_vehicle_border(y_tv[-1, 0], y_tv[-1, 1], y_tv[-1, 2], data_tv_cur[6],
                                        data_tv_cur[5], 1.2, hls='--', hcolor=hcolor_traj_tv, op=0.8, zorder=6)

        # Draw traj-hist (ego-vehicle)
        self.draw_traj(traj_tv_hist[:, 0:2], 1.0, '-', hcolor_hist_traj_tv, 1, zorder=3)

        # Draw arrows
        self.draw_vehicle_arrow(data_ov_cur, id_near, 20, hcolor_arrow_sel)
        self.draw_vehicle_arrow(data_ov_cur, id_rest, 20, hcolor_arrow_rest)
        self.draw_target_vehicle_arrow(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                       data_tv_cur[5], data_tv_cur[4], 20, hcolor_arrow_sel)

        # Draw borders
        self.draw_vehicle_border(data_ov_cur, id_near, 0.9, hcolor_border_ovsel)
        self.draw_vehicle_border(data_ov_cur, id_rest, 0.7, hcolor_border_ovrest)
        self.draw_target_vehicle_border(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                        data_tv_cur[5], 1.2, hcolor=hcolor_border_tv)

    def draw_ctrl_basic(self, data_ev_cur, data_ov_cur, id_near, id_rest, traj_sel_ev, traj_hist_ev, traj_ov_near,
                        size_ov_near, v_arrow=20):
        """
        Draws control-result (basic).
        """
        # Color setting
        hcolor_fill_ev = get_rgb("Crayola")
        hcolor_fill_ovsel, hcolor_fill_rest = get_rgb("Salmon"), get_rgb("Dark Gray")
        hcolor_arrow_sel, hcolor_arrow_rest = get_rgb("White"), get_rgb("Dark Gray")
        hcolor_border_ev = get_rgb("Han Blue")
        hcolor_border_ovsel, hcolor_border_ovrest = get_rgb("Dark Pastel Red"), get_rgb("Dim Gray")

        hcolor_traj_ev, hcolor_traj_ov = get_rgb("Han Blue"), get_rgb("Indian Red")
        hcolor_border_traj_ev = get_rgb("Corn Flower Blue")
        hcolor_hist_traj_ev = get_rgb("Dark Turquoise")

        # Draw other vehicle
        self.draw_vehicle_fill(data_ov_cur, id_near, hcolor_fill_ovsel, op=0.9)

        # Draw target vehicle
        self.draw_target_vehicle_fill(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                      data_ev_cur[5], hcolor_fill_ev, op=0.9)

        # Draw trajectories (other vehicle)
        self.draw_trajs(traj_ov_near, 0.9, '-', hcolor_traj_ov, op=0.5)
        self.draw_vehicle_fill_trajs_cgad(traj_ov_near, size_ov_near, 2, hcolor_fill_ovsel, hcolor_fill_rest, op=0.6)
        self.draw_vehicle_border_trajs(traj_ov_near, size_ov_near, 2, 0.5, hcolor_traj_ov, op=0.75)

        # Draw trajectory (target vehicle)
        self.draw_traj(traj_sel_ev[:, 0:2], 1.0, '--', hcolor_traj_ev, op=1.0, zorder=5)
        self.draw_vehicle_fill_traj_cgad(traj_sel_ev, data_ev_cur[6], data_ev_cur[5], 2, hcolor_fill_ev,
                                         hcolor_fill_rest, op=0.75, zorder=2)
        self.draw_vehicle_border_traj(traj_sel_ev, data_ev_cur[6], data_ev_cur[5], 2, 0.5, hcolor_border_traj_ev,
                                      op=1.0, zorder=4)

        self.draw_traj(traj_hist_ev[:, 0:2], 1.0, '-', hcolor_hist_traj_ev, op=1.0, zorder=5)

        # Draw arrows
        self.draw_vehicle_arrow(data_ov_cur, id_near, v_arrow, hcolor_arrow_sel)
        self.draw_vehicle_arrow(data_ov_cur, id_rest, v_arrow, hcolor_arrow_rest)
        self.draw_target_vehicle_arrow(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                       data_ev_cur[5], data_ev_cur[4], v_arrow, hcolor_arrow_sel)

        # Draw borders
        self.draw_vehicle_border(data_ov_cur, id_near, 0.9, hcolor_border_ovsel)
        self.draw_vehicle_border(data_ov_cur, id_rest, 0.7, hcolor_border_ovrest)
        self.draw_target_vehicle_border(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                        data_ev_cur[5], 1.2, hcolor=hcolor_border_ev)

    # DRAW-COSTMAP ----------------------------------------------------------------------------------------------------#
    def draw_track_costmap(self, zorder=5):
        """
        Draws track (costmap-version).
        """
        # pnts_poly_track: (list) points of track

        if self.mode_screen == 1:
            linewidth_outer, linewidth_inner = 1.2, 0.6
        else:
            linewidth_outer, linewidth_inner = 0.6, 0.4

        hcolor_outer_line = get_rgb("Khaki")
        hcolor_inner_line = get_rgb("White Smoke")

        # Plot track (outer)
        for nidx_seg in range(0, len(self.pnts_outer_border_track)):
            pnts_outer_seg = self.pnts_outer_border_track[nidx_seg]
            for nidx_lane in range(0, len(pnts_outer_seg)):
                pnts_outer_seg_sel = pnts_outer_seg[nidx_lane]
                self.axs.plot(pnts_outer_seg_sel[:, 0], pnts_outer_seg_sel[:, 1], linewidth=linewidth_outer,
                              color=hcolor_outer_line, zorder=zorder, solid_joinstyle='round', solid_capstyle='round')

        # Plot track (inner)
        for nidx_seg in range(0, len(self.pnts_inner_border_track)):
            pnts_inner_seg = self.pnts_inner_border_track[nidx_seg]
            for nidx_lane in range(0, len(pnts_inner_seg)):
                pnts_inner_lane = pnts_inner_seg[nidx_lane]
                self.axs.plot(pnts_inner_lane[:, 0], pnts_inner_lane[:, 1], linestyle="--", dashes=[5, 2],
                              linewidth=linewidth_inner, color=hcolor_inner_line, zorder=zorder,
                              solid_joinstyle='round', solid_capstyle='round')

        if self.plot_bk == 1:
            self.axs.set_facecolor("black")

    def draw_pred_costmap(self, data_tv_cur, data_ov_cur, id_near, id_rest, y_tv, y_tv_pred, traj_ov_near, size_ov_near,
                          traj_tv_hist, cost_tv_pred, map_x0, map_x1, map_z, hcolormap="viridis"):
        """
        Draws prediction (basic).
        """
        # Color setting
        hcolor_tv = get_rgb("Crayola")
        hcolor_ovsel, hcolor_ovrest = get_rgb("Crimson"), get_rgb("Dark Gray")

        hcolor_traj_tv = get_rgb("Dodger Blue")  # "Han Blue"
        hcolor_traj_ov = get_rgb("Indian Red")
        hcolor_hist_traj_tv = get_rgb("Dark Turquoise")

        if len(cost_tv_pred) > 0:
            cost_tv_pred_min, cost_tv_pred_max = min(cost_tv_pred), max(cost_tv_pred)
        else:
            cost_tv_pred_min, cost_tv_pred_max = 0.0, 1.0

        # Draw trajectories (other vehicle)
        self.draw_trajs(traj_ov_near, 0.9, '-', hcolor=hcolor_traj_ov, op=0.5)
        self.draw_vehicle_border_trajs(traj_ov_near, size_ov_near, 2, 0.5, hcolor=hcolor_traj_ov, op=0.75)

        # Draw trajectories (target vehicle)
        cmap = matplotlib.cm.get_cmap(hcolormap)
        for nidx_n in range(0, len(y_tv_pred)):
            y_tv_tmp = y_tv_pred[nidx_n]
            if len(cost_tv_pred) > 0:
                cmap_idx = (cost_tv_pred[nidx_n] - cost_tv_pred_min) / (cost_tv_pred_max - cost_tv_pred_min)
                cmap_sel = cmap(cmap_idx)
            else:
                cmap_sel = cmap((nidx_n - 1) / (len(y_tv_pred)))

            self.draw_traj(y_tv_tmp[:, 0:2], 2.5, '-', cmap_sel, op=0.62)

        y_tv = interpolate_traj(y_tv, alpha=5)
        self.draw_traj(y_tv[:, 0:2], 1.0, '--', hcolor_traj_tv, op=0.8, zorder=5)

        # Draw costmap
        plt.contourf(map_x0, map_x1, map_z, 30)

        # Draw traj-hist (ego-vehicle)
        self.draw_traj(traj_tv_hist[:, 0:2], 1.0, '-', hcolor_hist_traj_tv, 1, zorder=3)

        # Draw arrows
        self.draw_vehicle_arrow(data_ov_cur, id_near, 20, hcolor_ovsel)
        self.draw_vehicle_arrow(data_ov_cur, id_rest, 20, hcolor_ovrest)
        self.draw_target_vehicle_arrow(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                       data_tv_cur[5], data_tv_cur[4], 20, hcolor_tv)

        # Draw borders
        self.draw_vehicle_border(data_ov_cur, id_near, 0.9, hcolor_ovsel, zorder=10)
        self.draw_vehicle_border(data_ov_cur, id_rest, 0.7, hcolor_ovrest, zorder=10)
        self.draw_target_vehicle_border(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                        data_tv_cur[5], 1.2, hcolor=hcolor_tv, zorder=10)

        plt.xlim([min(map_x0), max(map_x0)])
        plt.ylim([min(map_x1), max(map_x1)])


if __name__ == '__main__':

    from src.utils_sim import *
    from core.SimTrack import *

    trackname = "US101"  # "US101", "I80", "highD_1"
    sim_track = SimTrack(trackname)

    # LOAD VEHICLE DATA -----------------------------------------------------------------------------------------------#
    #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
    if trackname == "US101":
        data_v = np.load("../data_vehicle/dv_ngsim_us101_1.npy")
    elif trackname == "I80":
        data_v = np.load("../data_vehicle/dv_ngsim_i80_2_s.npy")
    elif "highD" in trackname:
        text_split = trackname.split("_")
        track_num = text_split[-1]
        filename2read = "../data_vehicle/dv_highD_{:s}.npy".format(track_num)
        data_v = np.load(filename2read)
    else:
        data_v = np.load("../data_vehicle/dv_ngsim_us101_1.npy")

    # Target vehicle id
    id_unique = np.unique(data_v[:, -1])
    id_tv = id_unique[300]

    idx_tmp = np.where(data_v[:, -1] == id_tv)
    idx_tmp = idx_tmp[0]
    data_tv = data_v[idx_tmp, :]

    t_min, t_max = int(np.amin(data_tv[:, 0])), int(np.amax(data_tv[:, 0]))
    print("t_min: " + str(t_min) + ", t_max: " + str(t_max))

    # SET SCREEN ------------------------------------------------------------------------------------------------------#
    sim_screen_m = SimScreenMatplotlib(1)
    sim_screen_m.set_figure(12, 12)
    sim_screen_m.set_pnts_track_init(sim_track.pnts_poly_track, sim_track.pnts_outer_border_track,
                                     sim_track.pnts_inner_border_track)

    # GET CURRENT-INFO ------------------------------------------------------------------------------------------------#
    t_cur = int((t_min + t_max) / 2)

    idx_cur = np.where(data_v[:, 0] == t_cur)
    idx_cur = idx_cur[0]
    data_v_cur = data_v[idx_cur, :]

    idx_tv = np.where(data_v_cur[:, 9] == id_tv)
    idx_tv = idx_tv[0][0]  # indexes of target vehicle
    data_tv_cur = data_v_cur[idx_tv, :]

    idx_v_rest = np.setdiff1d(np.arange(0, data_v_cur.shape[0]), idx_tv)
    data_ov_cur = data_v_cur[idx_v_rest, :]

    # Get feature
    f_cur, id_near_cur, pnts_debug_f_ev, pnts_debug_f_ov, _, _ = get_feature(sim_track, data_tv_cur, data_ov_cur,
                                                                             use_intp=0)

    # DRAW ------------------------------------------------------------------------------------------------------------#
    sim_screen_m.set_pnt_range(data_tv_cur[1:3], [100, 100])

    sim_screen_m.draw_track()
    sim_screen_m.draw_vehicle_fill(data_ov_cur, id_near_cur, get_rgb("Crimson"))
    sim_screen_m.draw_target_vehicle_fill(data_tv_cur[1], data_tv_cur[2], data_tv_cur[3], data_tv_cur[6],
                                          data_tv_cur[5], get_rgb("Dodger Blue"))
    sim_screen_m.update_view_range()
    plt.show()
