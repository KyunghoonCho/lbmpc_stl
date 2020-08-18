from __future__ import print_function

IS_VIRTUALENV = 1

import numpy as np
import math

import pygame
import pygame.gfxdraw

if IS_VIRTUALENV:
    import cairocffi as cairo  # for virtual env
else:
    import cairo

import matplotlib
import matplotlib.cm

from PIL import Image
from src.utils import *
from src.get_rgb import *


class SimScreen(object):
    """
    SCREEN FOR SIMULATOR (PYGAME + PYCAIRO)
    """
    def __init__(self, sim_track, screen_mode=0, is_black=0, width_add=0, height_add=0):
        if screen_mode == 0:
            screen_alpha, screen_size = sim_track.screen_alpha_wide, sim_track.screen_size_wide
        else:
            screen_alpha, screen_size = sim_track.screen_alpha_narrow, sim_track.screen_size_narrow

        # Set screen settings
        self.screen_size = screen_size
        self.screen_size[0] = self.screen_size[0] + width_add
        self.screen_size[1] = self.screen_size[1] + height_add
        self.screen_alpha = screen_alpha
        self.screen_mode = screen_mode
        self.is_black = is_black

        # Set display
        pygame.init()  # Initialize the game engine
        pygame.display.set_caption("TRACK-SIM")
        self.pygame_screen = pygame.display.set_mode((self.screen_size[0], self.screen_size[1]), 0, 32)

        # Create raw surface data
        # data_surface_raw = np.empty(self.screen_size[0] * self.screen_size[1] * 4, dtype=np.int8)

        # Set surface (cairo)
        # stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, self.screen_size[0])
        # self.pycairo_surface = cairo.ImageSurface.create_for_data(data_surface_raw, cairo.FORMAT_ARGB32,
        #                                                           self.screen_size[0], self.screen_size[1], stride)

        self.pycairo_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.screen_size[0], self.screen_size[1])
        self.pycairo_surface.set_fallback_resolution(600, 600)

        # Set context (cairo)
        self.ctx = cairo.Context(self.pycairo_surface)

        # Pnts of range (screen)
        self.pnt_min, self.pnt_max = [], []

        # Pnts (track)
        self.pnts_poly_track, self.pnts_outer_border_track, self.pnts_inner_border_track = [], [], []
        self.pnts_goal = []

        # Pixel (track)
        self.pixel_poly_track, self.pixel_outer_border_track, self.pixel_inner_border_track = [], [], []

        # Pixel (vehicle)
        self.pixel_vehicle_others, self.pixel_vehicle_ego = [], []

        # Sub-screen (1)
        self.screen_size_sub = (screen_size[0]/3.0, screen_size[1]/3.0)
        self.screen_alpha_sub = 1.0
        self.pnt_min_sub, self.pnt_max_sub = [], []
        self.quadrant_number_sub = 2  # 1: upper-left, 2: upper-right, 3: lower-right, 4: lower-left

        # Sub-screen (2)
        self.screen_size_sub_2 = (screen_size[0] / 3.0, screen_size[1] / 3.0)
        self.screen_alpha_sub_2 = 1.0
        self.pnt_min_sub_2, self.pnt_max_sub_2 = [], []
        self.quadrant_number_sub_2 = 4  # 1: upper-left, 2: upper-right, 3: lower-right, 4: lower-left

        # Parameters w.r.t. pnts in track -----------------------------------------------------------------------------#
        # Set track-points
        self.set_pnts_track_init(sim_track.pnts_poly_track, sim_track.pnts_outer_border_track,
                                 sim_track.pnts_inner_border_track, sim_track.pnts_goal)

        # Set points range
        if screen_mode == 0:
            self.set_pnts_range(sim_track.pnt_min, sim_track.pnt_max)
        else:
            self.set_pnts_range_sub(sim_track.pnt_min, sim_track.pnt_max)

        # Set sub-screen
        if screen_mode == 1:
            self.set_screen_height_sub(screen_size[1] * sim_track.screen_sub_height_ratio)
            self.set_screen_alpha_sub()

        # Set display-clock -------------------------------------------------------------------------------------------#
        self.clock = pygame.time.Clock()  # Used to manage how fast the screen updates
        self.clock_rate = 60

    # SET PARAMETERS --------------------------------------------------------------------------------------------------#
    def set_screen_size(self, screen_size):
        """ Sets screen size.
        :param screen_size: (tuple) screen size (dim = 2)
        """
        self.screen_size = screen_size

    def set_screen_alpha(self, screen_alpha):
        """ Sets screen alpha.
        :param screen_alpha: screen ratio
        """
        self.screen_alpha = screen_alpha

    def set_pnts_range(self, pnt_min, pnt_max):
        """ Sets raange.
        :param pnt_min, pnt_max: point (dim = 2)
        """
        self.pnt_min = pnt_min
        self.pnt_max = pnt_max

    def set_pnts_range_wrt_mean(self, pnt_mean):
        """ Sets range w.r.t. mean.
        :param pnt_mean: point (dim = 2)
        """
        screen_size_tmp = np.array(self.screen_size)
        pnt_range = screen_size_tmp / self.screen_alpha
        self.pnt_min = pnt_mean - pnt_range / 2.0
        self.pnt_max = pnt_mean + pnt_range / 2.0

    def set_pnts_track_init(self, pnts_poly_track, pnts_outer_border_track, pnts_inner_border_track, pnts_goal):
        """ Sets rest of track-points. """
        self.pnts_poly_track = pnts_poly_track
        self.pnts_outer_border_track = pnts_outer_border_track
        self.pnts_inner_border_track = pnts_inner_border_track
        self.pnts_goal = pnts_goal

    # UTILS -----------------------------------------------------------------------------------------------------------#
    def convert2pixel(self, pnts):
        """ Converts points from Euclidean space to pixel space.
        :param pnts: points (dim = N x 2)
        """
        pnts = make_numpy_array(pnts, keep_1dim=False)
        num_pnts = pnts.shape[0]
        pnts_conv = np.zeros((num_pnts, 2), dtype=np.float64)

        pnts_conv[:, 0] = pnts[:, 0] - np.repeat(self.pnt_min[0], num_pnts, axis=0)
        pnts_conv[:, 1] = np.repeat(self.pnt_max[1], num_pnts, axis=0) - pnts[:, 1]

        pnts_conv = pnts_conv * self.screen_alpha
        # pnts_conv = np.round(pnts_conv, 0)

        return pnts_conv

    def update_pixel_track(self):
        """ Updates track points to pixel space. """
        self.pixel_poly_track = self.update_pixel_track_sub(self.pnts_poly_track)
        self.pixel_outer_border_track = self.update_pixel_track_sub(self.pnts_outer_border_track)
        self.pixel_inner_border_track = self.update_pixel_track_sub(self.pnts_inner_border_track)

    def update_pixel_track_sub(self, pnts_track):
        """ Updates (sub) track pixels. """
        pixel_track = []
        num_lane_seg = 0  # number of lane-segment
        for nidx_seg in range(0, len(pnts_track)):
            seg_sel = pnts_track[nidx_seg]

            pixel_seg = []
            for nidx_lane in range(0, len(seg_sel)):
                num_lane_seg = num_lane_seg + 1
                pnts_tmp = seg_sel[nidx_lane]
                pnts_conv_tmp = self.convert2pixel(pnts_tmp)
                pixel_seg.append(pnts_conv_tmp)

            pixel_track.append(pixel_seg)
        return pixel_track

    def snapCoords(self, x, y):
        # (xd, yd) = self.ctx.user_to_device(x, y)
        # out = (round(xd) + 0.5, round(yd) + 0.5)
        out = (x, y)
        return out

    def close(self):
        """ Closes pygame-window. """
        pygame.display.quit()

    def save_image(self, filename2save):
        """ Saves image. """
        # pygame.image.save(self.pygame_screen, filename2save)
        self.pycairo_surface.write_to_png(filename2save)

    # DRAW (BASIC) ----------------------------------------------------------------------------------------------------#
    def bgra_surf_to_rgba_string(self):
        """ Converts memoryview object to byte-array. """
        data_tmp = self.pycairo_surface.get_data()

        if IS_VIRTUALENV == 0:
            data_tmp = data_tmp.tobytes()  # Unable for virtualenv

        # Use PIL to get img
        img = Image.frombuffer('RGBA', (self.pycairo_surface.get_width(), self.pycairo_surface.get_height()), data_tmp,
                               'raw', 'BGRA', 0, 1)

        return img.tobytes('raw', 'RGBA', 0, 1)

    def draw_background(self, color_name):
        """ Draws background with specific color.
        :param color_name: color-name (string)
        """
        hcolor = get_rgb(color_name)

        self.ctx.move_to(0, 0)
        self.ctx.line_to(self.screen_size[0], 0)
        self.ctx.line_to(self.screen_size[0], self.screen_size[1])
        self.ctx.line_to(0, self.screen_size[1])
        self.ctx.line_to(0, 0)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
        self.ctx.fill()

    def draw_pnt(self, pnt, radius, hcolor, is_fill=True, op=1.0, sub_num=0):
        """ Draws point.
        :param pnt: point (dim = 1 x 2)
        :param radius: radius (float)
        :param hcolor: color (tuple)
        :param is_fill: filled circle (boolean)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        pnt = make_numpy_array(pnt, keep_1dim=True)
        pnt = pnt[0:2]
        pnt = np.reshape(pnt, (1, 2))

        # Convert to pixel space
        if sub_num == 0:
            pnt_sel_conv = self.convert2pixel(pnt)
        else:
            pnt_sel_conv = self.convert2pixel_sub(pnt, sub_num)

        pnt_sel_conv = self.snapCoords(pnt_sel_conv[0, 0], pnt_sel_conv[0, 1])
        self.ctx.arc(pnt_sel_conv[0], pnt_sel_conv[1], radius, 0, 2 * math.pi)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
        if is_fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def draw_pnts(self, pnts, radius, hcolor, is_fill=True, op=1.0, sub_num=0):
        """ Draw points.
        :param pnts: points (dim = N x 2)
        :param radius: radius (float)
        :param hcolor: color (tuple)
        :param is_fill: filled circle (boolean)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        pnts = make_numpy_array(pnts, keep_1dim=False)
        num_in = pnts.shape[0]

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, :]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            self.draw_pnt(pnt_sel, radius, hcolor, is_fill, op, sub_num=sub_num)

    def draw_pnts_cmap(self, pnts, radius, is_fill=True, op=1.0, sub_num=0):
        """ Draw points w.r.t. colormap.
        :param pnts: points (dim = N x 2)
        :param radius: radius (float)
        :param is_fill: filled circle (boolean)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        num_in = pnts.shape[0]
        cmap = matplotlib.cm.get_cmap("rainbow")

        for nidx_d in range(0, num_in):
            pnt_sel = pnts[nidx_d, 0:2]
            pnt_sel = np.reshape(pnt_sel, (1, 2))
            cmap_sel = cmap(nidx_d/num_in)
            self.draw_pnt(pnt_sel, radius, cmap_sel[0:3], is_fill, op, sub_num=sub_num)

    def draw_pnts_list(self, pnts_list, hcolor, radius, is_fill=True, op=1.0, sub_num=0):
        """ Draws points-list.
        :param pnts_list: list of points (dim = N x 2)
        :param hcolor: rgb color (tuple)
        :param radius: radius (float)
        :param is_fill: filled circle (boolean)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        for nidx_d in range(0, len(pnts_list)):
            pnts_sel = pnts_list[nidx_d]
            self.draw_pnts(pnts_sel[:, 0:2], radius, hcolor, is_fill, op, sub_num=sub_num)

    def draw_box(self, x, y, length, width, hlw, hcolor, sub_num=0):
        """ Draws box-points.
        :param x: position x (float)
        :param y: position y (float)
        :param length: length (float)
        :param width: width (float)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param sub_num: sub-plot number (int)
        """
        box_tmp = get_box_pnts(x, y, 0.0, length, width)
        box_tmp_0 = box_tmp[0, :]
        box_tmp = np.concatenate((box_tmp, box_tmp_0.reshape(1, -1)), axis=0)
        self.draw_traj(box_tmp[:, 0:2], hlw, hcolor, sub_num=sub_num)

    # DRAW (MAIN) -----------------------------------------------------------------------------------------------------#
    def draw_track(self, is_black=0):
        """ Draws track """
        if is_black == 1:
            hcolor_track = get_rgb("Dark Gray")
            hcolor_outer_line = get_rgb("White")
            hcolor_inner_line = get_rgb("White Smoke")
        else:
            hcolor_track = get_rgb("Gray")
            hcolor_outer_line = get_rgb("Very Dark Gray")
            hcolor_inner_line = get_rgb("White Smoke")

        if self.screen_mode == 0:
            linewidth_outer, linewidth_inner = 3, 1.25
        elif self.screen_mode == 1:
            linewidth_outer, linewidth_inner = 4, 1.5
        else:
            linewidth_outer, linewidth_inner = 2, 1

        self.update_pixel_track()

        # Plot track (polygon)
        for nidx_seg in range(0, len(self.pixel_poly_track)):
            pixel_poly_seg = self.pixel_poly_track[nidx_seg]

            # Plot lane-segment
            for nidx_lane in range(0, len(pixel_poly_seg)):
                idx_lane = len(pixel_poly_seg) - nidx_lane - 1
                # Pnts on lane-segment
                pixel_poly_lane = pixel_poly_seg[idx_lane]

                for nidx_pnt in range(0, pixel_poly_lane.shape[0]):
                    pixel_tmp = pixel_poly_lane[nidx_pnt, :]
                    pixel_tmp = self.snapCoords(pixel_tmp[0], pixel_tmp[1])
                    if nidx_pnt == 0:
                        self.ctx.move_to(pixel_tmp[0], pixel_tmp[1])
                    else:
                        self.ctx.line_to(pixel_tmp[0], pixel_tmp[1])

                pnt_0_tmp = pixel_poly_lane[0, :]
                pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
                self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

                # Plot (cairo)
                self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self.ctx.set_source_rgb(hcolor_track[0], hcolor_track[1], hcolor_track[2])
                self.ctx.fill_preserve()
                # self.ctx.set_source_rgb(0, 0, 0)
                # self.ctx.set_line_width(linewidth_outer)
            self.ctx.stroke()

        # Plot track (outer)
        for nidx_seg in range(0, len(self.pixel_outer_border_track)):
            pixel_outer_seg = self.pixel_outer_border_track[nidx_seg]
            for nidx_lane in range(0, len(pixel_outer_seg)):
                pixel_outer_lane = pixel_outer_seg[nidx_lane]

                # for nidx_pnt in range(0, pixel_outer_lane.shape[0]):
                #     pixel_tmp = pixel_outer_lane[nidx_pnt, :]
                #     pixel_tmp = self.snapCoords(pixel_tmp[0], pixel_tmp[1])
                #     if nidx_pnt == 0:
                #         self.ctx.move_to(pixel_tmp[0], pixel_tmp[1])
                #     else:
                #         self.ctx.line_to(pixel_tmp[0], pixel_tmp[1])

                len_pixel_tmp = pixel_outer_lane.shape[0]
                idx_tmp_cur = 0
                pixel_tmp = pixel_outer_lane[idx_tmp_cur, :]
                pixel_tmp = self.snapCoords(pixel_tmp[0], pixel_tmp[1])
                self.ctx.move_to(pixel_tmp[0], pixel_tmp[1])
                while idx_tmp_cur < (len_pixel_tmp - 3):
                    pixel_tmp1 = pixel_outer_lane[idx_tmp_cur + 1, :]
                    pixel_tmp2 = pixel_outer_lane[idx_tmp_cur + 2, :]
                    pixel_tmp3 = pixel_outer_lane[idx_tmp_cur + 3, :]
                    pixel_tmp1 = self.snapCoords(pixel_tmp1[0], pixel_tmp1[1])
                    pixel_tmp2 = self.snapCoords(pixel_tmp2[0], pixel_tmp2[1])
                    pixel_tmp3 = self.snapCoords(pixel_tmp3[0], pixel_tmp3[1])
                    self.ctx.curve_to(pixel_tmp1[0], pixel_tmp1[1], pixel_tmp2[0], pixel_tmp2[1], pixel_tmp3[0],
                                      pixel_tmp3[1])
                    idx_tmp_cur = idx_tmp_cur + 3

                if idx_tmp_cur < (len_pixel_tmp - 1):
                    pixel_tmp1 = pixel_outer_lane[idx_tmp_cur + 1, :]
                    if idx_tmp_cur < (len_pixel_tmp - 2):
                        pixel_tmp2 = pixel_outer_lane[idx_tmp_cur + 2, :]
                    else:
                        pixel_tmp2 = pixel_outer_lane[len_pixel_tmp - 1, :]

                    if idx_tmp_cur < (len_pixel_tmp - 3):
                        pixel_tmp3 = pixel_outer_lane[idx_tmp_cur + 3, :]
                    else:
                        pixel_tmp3 = pixel_outer_lane[len_pixel_tmp - 1, :]

                    pixel_tmp1 = self.snapCoords(pixel_tmp1[0], pixel_tmp1[1])
                    pixel_tmp2 = self.snapCoords(pixel_tmp2[0], pixel_tmp2[1])
                    pixel_tmp3 = self.snapCoords(pixel_tmp3[0], pixel_tmp3[1])

                    self.ctx.curve_to(pixel_tmp1[0], pixel_tmp1[1], pixel_tmp2[0], pixel_tmp2[1], pixel_tmp3[0],
                                      pixel_tmp3[1])

                # Plot (cairo)
                self.ctx.set_line_width(linewidth_outer)
                self.ctx.set_source_rgb(hcolor_outer_line[0], hcolor_outer_line[1], hcolor_outer_line[2])
                self.ctx.stroke()

        # Plot track (inner)
        for nidx_seg in range(0, len(self.pixel_inner_border_track)):
            pixel_inner_seg = self.pixel_inner_border_track[nidx_seg]
            for nidx_lane in range(0, len(pixel_inner_seg)):
                pixel_inner_lane = pixel_inner_seg[nidx_lane]

                for nidx_pnt in range(0, pixel_inner_lane.shape[0], 2):
                    pixel_cur_tmp = pixel_inner_lane[nidx_pnt, :]
                    pixel_cur_tmp = self.snapCoords(pixel_cur_tmp[0], pixel_cur_tmp[1])

                    if (nidx_pnt + 1) <= (pixel_inner_lane.shape[0] - 1):
                        pixel_next_tmp = pixel_inner_lane[nidx_pnt + 1, :]
                        pixel_next_tmp = self.snapCoords(pixel_next_tmp[0], pixel_next_tmp[1])

                        self.ctx.move_to(pixel_cur_tmp[0], pixel_cur_tmp[1])
                        self.ctx.line_to(pixel_next_tmp[0], pixel_next_tmp[1])

                        self.ctx.set_line_width(linewidth_inner)
                        self.ctx.set_source_rgb(hcolor_inner_line[0], hcolor_inner_line[1], hcolor_inner_line[2])
                        self.ctx.stroke()
                    else:
                        self.ctx.arc(pixel_cur_tmp[0], pixel_cur_tmp[1], 1, 0, 2 * math.pi)
                        self.ctx.set_source_rgb(hcolor_inner_line[0], hcolor_inner_line[1], hcolor_inner_line[2])
                        self.ctx.fill()

    def draw_traj(self, traj, hlw, hcolor, op=1.0, sub_num=0):
        """ Draws trajectory.
        :param traj: trajectory (dim = N x 2)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        # Convert to pixel space
        if sub_num == 0:
            traj_conv_tmp = self.convert2pixel(traj)
        else:
            traj_conv_tmp = self.convert2pixel_sub(traj, sub_num)

        # Plot (cairo)
        for nidx_pnt in range(0, traj_conv_tmp.shape[0]):
            pnt_tmp = traj_conv_tmp[nidx_pnt, :]
            pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        self.ctx.set_line_width(hlw)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.stroke()

    def draw_trajs(self, trajs, hlw, hcolor, op=1.0, sub_num=0):
        """ Draws trajectories.
        :param trajs: list of trajectories (dim = N x 2)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        for nidx_traj in range(0, len(trajs)):
            traj_sel = trajs[nidx_traj]
            self.draw_traj(traj_sel, hlw, hcolor, op, sub_num=sub_num)

    def draw_trajs_w_cost(self, trajs, costs, idx_invalid, cost_max, hlinewidth, hcolor_map="cool", op=1.0,
                          hcolor_reverse=False, sub_num=0):
        """ Draws trajectories w.r.t. cost.
        :param trajs: list of trajectories (dim = N x 2)
        :param costs: costs of trajectories (list)
        :param idx_invalid: invalid trajectory indexes (ndarray, list)
        :param cost_max: cost max (if 0, cost is scaled automatically) (float)
        :param hlinewidth: linewidth (float)
        :param hcolor_map: matplotlib-colormap name (string)
        :param op: opacity 0 ~ 1 (float)
        :param hcolor_reverse: matplotlib-colormap reverse (boolean)
        :param sub_num: sub-plot number (int)
        """
        if len(trajs) > 0:
            num_traj = len(trajs)
            if len(costs) > 0:
                costs = make_numpy_array(costs, keep_1dim=True)

                idx_sorted_tmp = np.argsort(costs)
                idx_sorted_tmp = np.flip(idx_sorted_tmp, axis=0)
                trajs = [trajs[idx_tmp] for idx_tmp in idx_sorted_tmp]
                costs = costs[idx_sorted_tmp]
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
                    min_cost_valid = np.amin(cost_valid)
                    max_cost_valid = np.amax(cost_valid)
                    if abs(max_cost_valid - min_cost_valid) < float(1e-4):
                        max_cost_valid, min_cost_valid = 1.0, 0.0
                else:
                    max_cost_valid, min_cost_valid = np.max(cost_valid), np.min(cost_valid)
            else:
                max_cost_valid, min_cost_valid = np.max(costs), np.min(costs)

            cmap = matplotlib.cm.get_cmap(hcolor_map)
            for nidx_traj in range(0, idx_valid.shape[0]):  # Plot valid trajectories
                idx_sel = idx_valid[nidx_traj]
                traj_sel = trajs[idx_sel]

                cmap_ratio = (costs[idx_sel] - min_cost_valid) / (max_cost_valid - min_cost_valid)
                cmap_ratio = 1.0 - cmap_ratio if hcolor_reverse else cmap_ratio

                cmap_ratio = min(max(cmap_ratio, 0.0), 1.0)
                cmap_tmp = cmap(cmap_ratio)
                self.draw_traj(traj_sel[:, 0:2], hlinewidth, cmap_tmp, op, sub_num=sub_num)

            # for nidx_traj in range(0, idx_invalid.shape[0]):  # Plot invalid trajectories
            #     idx_sel = idx_invalid[nidx_traj]
            #     traj_sel = traj_list[idx_sel]
            #     self.draw_trajectory(traj_sel[:, 0:2], hlinewidth, get_rgb("Dark Slate Blue"), op)

    def draw_traj_dashed(self, traj, hlw, hcolor, op=1.0, sub_num=0):
        """ Draws dashed trajectory (sub).
        :param traj: trajectory (dim = N x 2)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        # Convert to pixel space
        if sub_num == 0:
            traj_conv_tmp = self.convert2pixel(traj)
        else:
            traj_conv_tmp = self.convert2pixel_sub(traj, sub_num)

        # Plot (cairo)
        for nidx_pnt in range(0, traj_conv_tmp.shape[0], 1):
            pnt_tmp = traj_conv_tmp[nidx_pnt, :]
            pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

            if nidx_pnt % 2 == 0:
                self.ctx.set_line_width(hlw)
                # self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])
                self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
                self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self.ctx.stroke()

    def draw_vehicle_fill_traj_cgad(self, traj, rx, ry, stepsize, hcolor1, hcolor2, op=1.0, sub_num=0):
        """ Draws vehicle fill trajectory with color gradient.
        :param traj: trajectory (dim = N x 3)
        :param rx: vehicle size (x) (float)
        :param ry: vehicle size (y) (float)
        :param stepsize: step-size (int)
        :param hcolor1: start rgb color (tuple)
        :param hcolor2: end rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
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
                self.draw_target_vehicle_fill(pnt_sel[0], pnt_sel[1], pnt_sel[2], rx, ry, hcolor_sel, op=op,
                                              sub_num=sub_num)

    def draw_vehicle_fill_trajs_cgad(self, trajs, sizes, stepsize, hcolor1, hcolor2, op=1.0, sub_num=0):
        """ Draws vehicle fill trajectories with color gradient.
        :param trajs: list of trajectories (ndarray, dim = N x 3)
        :param sizes: list of sizes rx, ry (ndarray, dim = N x 2)
        :param stepsize: step-size (int)
        :param hlinewidth: linewidth (float)
        :param hcolor1: start rgb color (tuple)
        :param hcolor2: end rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        for nidx_traj in range(0, len(trajs)):
            traj_sel, size_sel = trajs[nidx_traj], sizes[nidx_traj, :]
            rx_sel, ry_sel = size_sel[0], size_sel[1]
            self.draw_vehicle_fill_traj_cgad(traj_sel, rx_sel, ry_sel, stepsize, hcolor1, hcolor2, op=op,
                                             sub_num=sub_num)

    def draw_vehicle_border_traj(self, traj, rx, ry, stepsize, hlinewidth, hcolor, op=1.0, sub_num=0):
        """ Draws vehicle border trajectory.
        :param traj: trajectory (ndarray, dim = N x 3)
        :param rx: vehicle size (x) (float)
        :param ry: vehicle size (y) (float)
        :param stepsize: step-size (int)
        :param hlinewidth: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        for nidx_t in range(stepsize, traj.shape[0], stepsize):
            pnt_sel = traj[nidx_t, :]

            if ~np.isnan(pnt_sel[0]):
                self.draw_target_vehicle_border(pnt_sel[0], pnt_sel[1], pnt_sel[2], rx, ry, hlinewidth, hcolor, op=op,
                                                sub_num=sub_num)

    def draw_vehicle_border_trajs(self, trajs, sizes, stepsize, hlinewidth, hcolor, op=1.0, sub_num=0):
        """ Draws vehicle border trajectories.
        :param trajs: list of trajectories (ndarray, dim = N x 3)
        :param sizes: list of sizes rx, ry (ndarray, dim = N x 2)
        :param stepsize: step-size (int)
        :param hlinewidth: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        for nidx_traj in range(0, len(trajs)):
            traj_sel = trajs[nidx_traj]
            size_sel = sizes[nidx_traj, :]
            rx_sel = size_sel[0]
            ry_sel = size_sel[1]
            self.draw_vehicle_border_traj(traj_sel, rx_sel, ry_sel, stepsize, hlinewidth, hcolor, op=op, sub_num=sub_num)

    def draw_vehicle_fill(self, data_v, ids, hcolor, op=1.0, sub_num=0):
        """ Draws vehicles (fill).
        :param data_v: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param ids: vehicle ids (list)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        if len(data_v) > 0:
            data_v = make_numpy_array(data_v, keep_1dim=False)

            num_v = data_v.shape[0]
            for nidx_n in range(0, num_v):
                # Get vehicle-data
                #       structure: t x y theta v length width tag_segment tag_lane id (dim = 10, width > length)
                data_v_tmp = data_v[nidx_n, :]

                # Get polygon-points
                pnts_v_tmp = get_pnts_carshape(data_v_tmp[1], data_v_tmp[2], data_v_tmp[3], data_v_tmp[6], data_v_tmp[5])

                # Set (fill) color
                if np.isin(data_v_tmp[-1], ids):
                    cmap_vehicle = hcolor
                else:
                    cmap_vehicle = get_rgb("Light Gray")

                # Convert to pixel space
                if sub_num == 0:
                    pnts_v_pixel = self.convert2pixel(pnts_v_tmp)
                else:
                    pnts_v_pixel = self.convert2pixel_sub(pnts_v_tmp, sub_num)

                # Plot vehicle
                for nidx_pnt in range(0, pnts_v_pixel.shape[0]):
                    pnt_tmp = pnts_v_pixel[nidx_pnt, :]
                    pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
                    if nidx_pnt == 0:
                        self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                    else:
                        self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

                pnt_0_tmp = pnts_v_pixel[0, :]
                pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
                self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

                # Plot (cairo)
                self.ctx.set_line_width(1)
                self.ctx.set_source_rgba(cmap_vehicle[0], cmap_vehicle[1], cmap_vehicle[2], op)
                self.ctx.fill_preserve()
                # self.ctx.set_source_rgb(0, 0, 0)
                # self.ctx.set_line_width(1)
                self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self.ctx.stroke()

    def draw_target_vehicle_fill(self, x, y, theta, rx, ry, hcolor, op=1.0, sub_num=0):
        """ Draws target vehicle (fill).
        :param x: position-x (float)
        :param y: position-y (float)
        :param theta: heading (rad) (float)
        :param rx: length (float)
        :param ry: width (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        # Get polygon-points
        pnts_v_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        # Convert to pixel space
        if sub_num == 0:
            pnts_v_pixel = self.convert2pixel(pnts_v_tmp)
        else:
            pnts_v_pixel = self.convert2pixel_sub(pnts_v_tmp, sub_num)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_v_pixel.shape[0]):
            pnt_tmp = pnts_v_pixel[nidx_pnt, :]
            pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_v_pixel[0, :]
        pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
        self.ctx.fill_preserve()
        # self.ctx.set_source_rgb(0, 0, 0)
        # self.ctx.set_line_width(1)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.stroke()

    def draw_vehicle_arrow(self, data_v, ids, lv_max, hcolor, op=1.0, sub_num=0):
        """ Draws vehicle arrow.
        :param data_v: vehicle data [t x y theta v length width segment lane id] (ndarray, dim = N x 10)
        :param ids: vehicle ids (list)
        :param lv_max: maximum linear-velocity (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
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
                    data_vehicle_sel = data_v[idx_tmp, :]
                    data_vehicle_sel = data_vehicle_sel.reshape(-1)
                    self.draw_target_vehicle_arrow(data_vehicle_sel[1], data_vehicle_sel[2], data_vehicle_sel[3],
                                                   data_vehicle_sel[6], data_vehicle_sel[5], data_vehicle_sel[4],
                                                   lv_max, hcolor, op, sub_num=sub_num)

    def draw_target_vehicle_arrow(self, x, y, theta, rx, ry, lv, lv_ref, hcolor, op=1.0, sub_num=0):
        """ Draws target vehicle arrow.
        :param x: position x (float)
        :param y: position y (float)
        :param theta: heading (rad) (float)
        :param rx: length (float)
        :param ry: width (float)
        :param lv: linear velocity (float)
        :param lv_ref: reference linear velocity (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        ratio_lv = 0.15 + (abs(lv) / lv_ref) * 0.85
        ratio_lv = min(ratio_lv, 1.0)

        ax, ay = ratio_lv * rx * 0.4, ry * 0.15
        bx, by = ratio_lv * rx * 0.15, ry * 0.15

        # Get polygon-points
        pnts_v_tmp = get_pnts_arrow(x, y, theta, ax, ay, bx, by)

        # Convert to pixel space
        if sub_num == 0:
            pnts_v_pixel = self.convert2pixel(pnts_v_tmp)
        else:
            pnts_v_pixel = self.convert2pixel_sub(pnts_v_tmp, sub_num)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_v_pixel.shape[0]):
            pnt_tmp = pnts_v_pixel[nidx_pnt, :]
            pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_v_pixel[0, :]
        pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.fill()

    def draw_vehicle_border(self, data_v, ids, hlw, hcolor, op=1.0, sub_num=0):
        """ Draws vehicle border.
        :param data_v: vehicle data [t x y theta v length width segment lane id] (dim = N x 10)
        :param ids: vehicle ids (list)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        if len(data_v) > 0:
            data_v = make_numpy_array(data_v, keep_1dim=False)
            len_id = len(ids)

            for nidx_d in range(0, len_id):
                id_tmp = ids[nidx_d]
                # if id_tmp == -1:
                #     continue

                idx_found_ = np.where(data_v[:, -1] == id_tmp)
                idx_found = idx_found_[0]

                if len(idx_found) > 0:
                    data_vehicle_sel = data_v[idx_found[0], :]
                    self.draw_target_vehicle_border(data_vehicle_sel[1], data_vehicle_sel[2], data_vehicle_sel[3],
                                                    data_vehicle_sel[6], data_vehicle_sel[5], hlw, hcolor, op=op,
                                                    sub_num=sub_num)

    def draw_target_vehicle_border(self, x, y, theta, rx, ry, hlw, hcolor, op=1.0, sub_num=0):
        """ Draws target vehicle border.
        :param x: position x (float)
        :param y: position y (float)
        :param theta: heading (rad) (float)
        :param rx: length (float)
        :param ry: width (float)
        :param hlw: linewidth (float)
        :param hcolor: rgb color (tuple)
        :param op: opacity 0 ~ 1 (float)
        :param sub_num: sub-plot number (int)
        """
        # Get polygon-points
        pnts_vehicle_tmp = get_pnts_carshape(x, y, theta, rx, ry)

        # Convert to pixel space
        if sub_num == 0:
            pnts_vehicle_pixel = self.convert2pixel(pnts_vehicle_tmp)
        else:
            pnts_vehicle_pixel = self.convert2pixel_sub(pnts_vehicle_tmp, sub_num)

        # Plot vehicle
        for nidx_pnt in range(0, pnts_vehicle_pixel.shape[0]):
            pnt_tmp = pnts_vehicle_pixel[nidx_pnt, :]
            pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
            if nidx_pnt == 0:
                self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
            else:
                self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

        pnt_0_tmp = pnts_vehicle_pixel[0, :]
        pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
        self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

        # Plot (cairo)
        self.ctx.set_line_width(hlw)
        self.ctx.set_source_rgba(hcolor[0], hcolor[1], hcolor[2], op)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.stroke()

    def draw_text(self, pnt, txt, fontsize, hcolor, is_bold=0, sub_num=0):
        """ Draws target vehicle border.
        :param pnt: point (dim = 2, 1 x 2)
        :param txt: text (string)
        :param fontsize: font-size (float)
        :param hcolor: rgb color (tuple)
        :param is_bold: whether font is bold (boolean)
        :param sub_num: sub-plot number (int)
        """
        pnt = make_numpy_array(pnt, keep_1dim=True)
        pnt = pnt[0:2]
        pnt = np.reshape(pnt, (1, 2))

        # Convert to pixel space
        if sub_num == 0:
            pnt_sel_conv = self.convert2pixel(pnt)
        else:
            pnt_sel_conv = self.convert2pixel_sub(pnt, sub_num)
        pnt_sel_conv = self.snapCoords(pnt_sel_conv[0, 0], pnt_sel_conv[0, 1])

        # Set color
        self.ctx.set_source_rgb(hcolor[0], hcolor[1], hcolor[2])

        # Is bold
        if is_bold:
            cairo_bold = cairo.FONT_WEIGHT_BOLD
        else:
            cairo_bold = cairo.FONT_WEIGHT_NORMAL

        self.ctx.select_font_face("Courier", cairo.FONT_SLANT_NORMAL, cairo_bold)
        self.ctx.set_font_size(fontsize)
        self.ctx.move_to(pnt_sel_conv[0], pnt_sel_conv[1])
        self.ctx.show_text(txt)

    def draw_texts_nearby(self, data_ev, data_ov, id_near, fontsize, hcolor, is_bold=0, sub_num=0):
        """
        Draws nearby vehicle txt-info.
        :param data_ev: ego vehicle data (dim = 10)
        :param data_ov: other vehicle data (dim = N x 10)
        :param id_near: near vehicle ids (id_lf, id_lr, id_rf, id_rr, id_cf, id_cr) (dim = 6)
        :param fontsize: font-size (float)
        :param hcolor: rgb color (tuple)
        :param is_bold: whether font is bold (boolean)
        :param sub_num: sub-plot number (int)
        """
        self.draw_text(data_ev[1:3], 'ev', fontsize, hcolor, is_bold=is_bold, sub_num=sub_num)
        txts_ov = ['lf', 'lr', 'rf', 'rr', 'cf', 'cr']
        for nidx_ov in range(0, 6):
            if id_near[nidx_ov] == -1:
                continue
            else:
                idx_found_tmp = np.where(data_ov[:, -1] == id_near[nidx_ov])
                idx_found_tmp = idx_found_tmp[0]
                pnt_ov_sel = data_ov[idx_found_tmp, 1:3]
                txts_ov_sel = txts_ov[nidx_ov]
                self.draw_text(pnt_ov_sel, txts_ov_sel, fontsize, hcolor, is_bold=is_bold, sub_num=sub_num)

    # -----------------------------------------------------------------------------------------------------------------#
    # SUB-SCREEN (UPPER-RIGHT QUADRANT) -------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    # SET PARAMETERS (SUB) --------------------------------------------------------------------------------------------#
    def set_screen_size_sub(self, screen_size, sub_num=1):
        """ Sets (sub) screen size.
        :param screen_size: screen size (tuple, dim = 2)
        :param sub_num: sub-plot number (int)
        """
        if sub_num == 1:
            self.screen_size_sub = screen_size
        else:
            self.screen_size_sub_2 = screen_size

    def set_screen_height_sub(self, height, sub_num=1):
        """ Sets (sub) screen height.
        :param height: screen height (float)
        :param sub_num: sub-plot number (int)
        """

        if sub_num == 1:
            ratio_hegiht2width = (self.pnt_max_sub[0] - self.pnt_min_sub[0]) / (
                        self.pnt_max_sub[1] - self.pnt_min_sub[1])
            self.screen_size_sub = [ratio_hegiht2width * height, height]
        else:
            ratio_hegiht2width = (self.pnt_max_sub_2[0] - self.pnt_min_sub_2[0]) / (
                    self.pnt_max_sub_2[1] - self.pnt_min_sub_2[1])
            self.screen_size_sub_2 = [ratio_hegiht2width * height, height]

    def set_pnts_range_sub(self, pnt_min, pnt_max, sub_num=1):
        """ Sets (sub) range.
        :param pnt_min: min point (list or ndarray, dim = 2)
        :param pnt_max: max point (list or ndarray, dim = 2)
        :param sub_num: sub-plot number (int)
        """

        if sub_num == 1:
            self.pnt_min_sub = pnt_min
            self.pnt_max_sub = pnt_max
        else:
            self.pnt_min_sub_2 = pnt_min
            self.pnt_max_sub_2 = pnt_max

    def set_screen_alpha_sub(self, sub_num=1):
        """ Sets (sub) screen alpha.
        :param sub_num: sub-plot number (int)
        """

        if sub_num == 1:
            if len(self.pnt_min_sub) > 0 and len(self.pnt_max_sub) > 0:
                alpha1 = self.screen_size_sub[0] / (self.pnt_max_sub[0] - self.pnt_min_sub[0])
                alpha2 = self.screen_size_sub[1] / (self.pnt_max_sub[1] - self.pnt_min_sub[1])
            else:
                alpha1, alpha2 = 1.0, 1.0
            self.screen_alpha_sub = min(alpha1, alpha2)
        else:
            if len(self.pnt_min_sub_2) > 0 and len(self.pnt_max_sub_2) > 0:
                alpha1 = self.screen_size_sub_2[0] / (self.pnt_max_sub_2[0] - self.pnt_min_sub_2[0])
                alpha2 = self.screen_size_sub_2[1] / (self.pnt_max_sub_2[1] - self.pnt_min_sub_2[1])
            else:
                alpha1, alpha2 = 1.0, 1.0
            self.screen_alpha_sub_2 = min(alpha1, alpha2)

    def set_pnts_range_wrt_mean_sub(self, pnt_mean, sub_num=1):
        """ Sets range w.r.t. mean (sub).
        :param pnt_mean: point (dim = 2)
        :param sub_num: sub-plot number (int)
        """
        if sub_num == 1:
            screen_alpha_sub = self.screen_alpha_sub
            screen_size_sub = self.screen_size_sub
        else:
            screen_alpha_sub = self.screen_alpha_sub_2
            screen_size_sub = self.screen_size_sub_2

        screen_size_tmp = np.array(screen_size_sub)
        pnt_range = screen_size_tmp / screen_alpha_sub

        if sub_num == 1:
            self.pnt_min_sub = pnt_mean - pnt_range / 2.0
            self.pnt_max_sub = pnt_mean + pnt_range / 2.0
        else:
            self.pnt_min_sub_2 = pnt_mean - pnt_range / 2.0
            self.pnt_max_sub_2 = pnt_mean + pnt_range / 2.0

    def set_quadrant_number_sub(self, quadrant_number, sub_num=1):
        """ Sets (sub) quadrant number.
        :param quadrant_number: 1, 2, 3, 4
        :param sub_num: sub-plot number (int)
        """
        if sub_num == 1:
            self.quadrant_number_sub = quadrant_number
        else:
            self.quadrant_number_sub_2 = quadrant_number

    # UTILS (SUB) -----------------------------------------------------------------------------------------------------#
    def convert2pixel_sub(self, pnts, sub_num=1):
        """ Converts points from Euclidean space to pixel space (sub).
        :param pnts: points (dim = N x 2)
        :param sub_num: sub-plot number (int)
        """
        if sub_num == 1:
            pnt_min_sub, pnt_max_sub = self.pnt_min_sub, self.pnt_max_sub
            screen_alpha_sub = self.screen_alpha_sub
            screen_size_sub = self.screen_size_sub
            quadrant_number_sub = self.quadrant_number_sub
        else:
            pnt_min_sub, pnt_max_sub = self.pnt_min_sub_2, self.pnt_max_sub_2
            screen_alpha_sub = self.screen_alpha_sub_2
            screen_size_sub = self.screen_size_sub_2
            quadrant_number_sub = self.quadrant_number_sub_2

        pnts = make_numpy_array(pnts, keep_1dim=False)

        num_pnts = pnts.shape[0]
        pnts_conv = np.zeros((num_pnts, 2), dtype=np.float32)

        pnts_conv[:, 0] = pnts[:, 0] - np.repeat(pnt_min_sub[0], num_pnts, axis=0)
        pnts_conv[:, 1] = np.repeat(pnt_max_sub[1], num_pnts, axis=0) - pnts[:, 1]

        pnts_conv = pnts_conv * screen_alpha_sub

        ratio_hegiht2width = (pnt_max_sub[0] - pnt_min_sub[0]) / (pnt_max_sub[1] - pnt_min_sub[1])
        width_sub_tmp = screen_size_sub[0] - (ratio_hegiht2width * screen_size_sub[1])

        if quadrant_number_sub == 1:
            pnt_move = [1, 1]
        elif quadrant_number_sub == 2:
            pnt_move = [self.screen_size[0] - screen_size_sub[0] + width_sub_tmp - 1, 1]
        elif quadrant_number_sub == 3:
            pnt_move = [self.screen_size[0] - screen_size_sub[0] + width_sub_tmp - 1,
                        self.screen_size[1] - screen_size_sub[1] - 1]
        elif quadrant_number_sub == 4:
            pnt_move = [1, self.screen_size[1] - screen_size_sub[1] - 1]
        else:
            pnt_move = [0, 0]

        pnts_conv[:, 0] = pnts_conv[:, 0] + pnt_move[0]
        pnts_conv[:, 1] = pnts_conv[:, 1] + pnt_move[1]

        return pnts_conv

    # DRAW (SUB) ------------------------------------------------------------------------------------------------------#
    def draw_background_sub(self, hcolor, hlw, sub_num=1):
        """ Draws background (sub).
        :param hcolor: rgb color (tuple)
        :param hlw: linewidth (float)
        :param sub_num: sub-plot number (int)
        """
        if sub_num == 1:
            pnt_min_sub, pnt_max_sub = self.pnt_min_sub, self.pnt_max_sub
        else:
            pnt_min_sub, pnt_max_sub = self.pnt_min_sub_2, self.pnt_max_sub_2

        traj = np.zeros((5, 2), dtype=np.float32)
        traj[0, :] = [pnt_min_sub[0], pnt_min_sub[1]]
        traj[1, :] = [pnt_max_sub[0], pnt_min_sub[1]]
        traj[2, :] = [pnt_max_sub[0], pnt_max_sub[1]]
        traj[3, :] = [pnt_min_sub[0], pnt_max_sub[1]]
        traj[4, :] = [pnt_min_sub[0], pnt_min_sub[1]]
        self.draw_traj(traj, hlw, hcolor, sub_num=sub_num)

    def draw_track_sub(self, is_black=0, sub_num=1):
        """ Draws track (sub).
        :param is_black: whether to plot black theme (boolean)
        :param sub_num: sub-plot number (int)
        """
        if is_black == 1:
            hcolor_track = get_rgb("Dark Gray")
        else:
            hcolor_track = get_rgb("Gray")

        # Convert to pixel space
        pnts_pixel_track = []
        num_lane_seg = 0  # number of lane-segment
        for nidx_seg in range(0, len(self.pnts_poly_track)):
            seg_sel = self.pnts_poly_track[nidx_seg]

            pnts_pixel_seg = []
            for nidx_lane in range(0, len(seg_sel)):
                num_lane_seg = num_lane_seg + 1
                pnts_tmp = seg_sel[nidx_lane]
                pnts_conv_tmp = self.convert2pixel_sub(pnts_tmp, sub_num)
                pnts_pixel_seg.append(pnts_conv_tmp)

            pnts_pixel_track.append(pnts_pixel_seg)

        # Plot track
        for nidx_seg in range(0, len(pnts_pixel_track)):
            pnts_pixel_seg = pnts_pixel_track[nidx_seg]

            # Plot lane-segment
            for nidx_lane in range(0, len(pnts_pixel_seg)):
                # Pnts on lane-segment
                pnts_pixel_lane = pnts_pixel_seg[nidx_lane]

                for nidx_pnt in range(0, pnts_pixel_lane.shape[0]):
                    pnt_tmp = pnts_pixel_lane[nidx_pnt, :]
                    pnt_tmp = self.snapCoords(pnt_tmp[0], pnt_tmp[1])
                    if nidx_pnt == 0:
                        self.ctx.move_to(pnt_tmp[0], pnt_tmp[1])
                    else:
                        self.ctx.line_to(pnt_tmp[0], pnt_tmp[1])

                pnt_0_tmp = pnts_pixel_lane[0, :]
                pnt_0_tmp = self.snapCoords(pnt_0_tmp[0], pnt_0_tmp[1])
                self.ctx.line_to(pnt_0_tmp[0], pnt_0_tmp[1])

                # Plot (cairo)
                self.ctx.set_line_width(0.3)
                self.ctx.set_source_rgb(hcolor_track[0], hcolor_track[1], hcolor_track[2])
                self.ctx.fill_preserve()
                self.ctx.set_source_rgb(0, 0, 0)
                self.ctx.set_line_width(0.3)
                self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                self.ctx.stroke()

    # -----------------------------------------------------------------------------------------------------------------#
    # DRAW-BASIC ------------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    def draw_basic(self, pnt_ref):
        """ Draws track.
        :param pnt_ref: reference point (x, y) (ndarray)
        """
        pygame.event.get()

        # Clear screen
        if self.is_black == 1:
            txt_backcolor = "Black"
        else:
            txt_backcolor = "White Smoke"
        self.draw_background(txt_backcolor)

        # Set middle point of window
        if self.screen_mode == 1:
            pnt_m_plot = np.array([pnt_ref[0], pnt_ref[1]], dtype=np.float32)
            self.set_pnts_range_wrt_mean(pnt_m_plot)

        self.draw_track(is_black=self.is_black)  # Draw track

    def draw_basic_sub(self):
        """ Draws track (sub). """
        if self.is_black == 1:
            txt_subtrackcolor = "White Smoke"
        else:
            txt_subtrackcolor = "Black"

        self.draw_background_sub(get_rgb(txt_subtrackcolor), 1.5, sub_num=1)
        self.draw_track_sub(is_black=self.is_black, sub_num=1)
        if len(self.pnts_goal) > 0:
            self.draw_pnts(self.pnts_goal, 1.0, get_rgb("Yellow"), sub_num=1)

    def draw_basic_sub_mpcstl(self, pnt_ref, sim_mpcstl, traj_computed, rmin_l_d=0, rmin_l_u=0, sub_num=2):
        """ Draws track (sub, mpc-stl). """
        if self.is_black == 1:
            txt_subtrackcolor = "White Smoke"
        else:
            txt_subtrackcolor = "Black"

        # Set basic setting
        pnt_m_plot = np.array([pnt_ref[0], pnt_ref[1]], dtype=np.float32)
        self.set_pnts_range_wrt_mean_sub(pnt_m_plot, sub_num=sub_num)
        self.draw_background_sub(get_rgb(txt_subtrackcolor), 1.5, sub_num=sub_num)

        # Get data from sim_mpcstl
        traj_ov_cf = sim_mpcstl.traj_ov_cf[:, 0:2]
        size_ov_cf = sim_mpcstl.size_ov_cf
        traj_ov = sim_mpcstl.traj_ov
        size_ov = sim_mpcstl.size_ov
        xinit_conv = sim_mpcstl.xinit_conv[0:2]
        xgoal_conv = sim_mpcstl.xgoal_conv[0:2]
        cp_l_d, cp_l_u = sim_mpcstl.cp_l_d, sim_mpcstl.cp_l_u
        rx, ry = sim_mpcstl.rx, sim_mpcstl.ry
        id_near = sim_mpcstl.id_ov_near
        id_cf, id_rest = id_near[4], id_near[[0, 1, 2, 3, 5]]

        pnt_min, pnt_max = self.pnt_min_sub_2, self.pnt_max_sub_2

        # Set range
        traj_computed = set_traj_in_range(traj_computed[:, 0:2], pnt_min, pnt_max)
        traj_ov_cf = set_traj_in_range(traj_ov_cf[:, 0:2], pnt_min, pnt_max)
        for nidx_ov in range(0, 5):
            traj_ov_sel = traj_ov[nidx_ov]
            traj_ov_sel_out = set_traj_in_range(traj_ov_sel[:, 0:2], pnt_min, pnt_max)
            traj_ov[nidx_ov] = traj_ov_sel_out

        # Draw lines -----------------------------------------------------------------------------#
        xtmp = np.arange(start=pnt_min[0], stop=pnt_max[0], step=0.5)
        len_xtmp = xtmp.shape[0]
        y_lower = (cp_l_d[1]) * np.ones((len_xtmp,), dtype=np.float32)
        y_upper = (cp_l_u[1]) * np.ones((len_xtmp,), dtype=np.float32)
        ytraj_lower = np.concatenate((xtmp.reshape(-1, 1), y_lower.reshape(-1, 1)), axis=1)
        ytraj_upper = np.concatenate((xtmp.reshape(-1, 1), y_upper.reshape(-1, 1)), axis=1)
        self.draw_traj_dashed(ytraj_lower, 3, sim_mpcstl.hcolor_lane_d, sub_num=sub_num)
        self.draw_traj_dashed(ytraj_upper, 3, sim_mpcstl.hcolor_lane_u, sub_num=sub_num)

        y_lower_mod = (cp_l_d[1] + rmin_l_d) * np.ones((len_xtmp,), dtype=np.float32)
        y_upper_mod = (cp_l_u[1] - rmin_l_u) * np.ones((len_xtmp,), dtype=np.float32)
        ytraj_lower_mod = np.concatenate((xtmp.reshape(-1, 1), y_lower_mod.reshape(-1, 1)), axis=1)
        ytraj_upper_mod = np.concatenate((xtmp.reshape(-1, 1), y_upper_mod.reshape(-1, 1)), axis=1)
        self.draw_traj(ytraj_lower_mod, 1.5, get_color_blurred(sim_mpcstl.hcolor_lane_d, 0.8), op=0.8, sub_num=sub_num)
        self.draw_traj(ytraj_upper_mod, 1.5, get_color_blurred(sim_mpcstl.hcolor_lane_u, 0.8), op=0.8, sub_num=sub_num)

        # Draw other vehicles ([id_lf, id_lr, id_rf, id_rr, id_cr]) ------------------------------#
        color_ov = [get_rgb('Crimson'), get_rgb('Dark Red'), get_rgb('Lawn Green'), get_rgb('Forest Green'),
                    get_rgb('Dark Cyan')]
        for nidx_ov in range(0, 5):
            traj_ov_sel = traj_ov[nidx_ov]
            size_ov_sel = size_ov[nidx_ov]
            hcolor_ov_sel = color_ov[nidx_ov]
            if len(traj_ov_sel) > 0:
                self.draw_traj(traj_ov_sel[:, 0:2], 2.5, hcolor_ov_sel, sub_num=sub_num)
                self.draw_pnt(traj_ov_sel[0, 0:2], 5, hcolor_ov_sel, is_fill=True, sub_num=sub_num)
                self.draw_box(traj_ov_sel[0, 0], traj_ov_sel[0, 1], size_ov_sel[0, 0], size_ov_sel[0, 1], 3.0,
                              hcolor_ov_sel, sub_num=sub_num)
                self.draw_box(traj_ov_sel[-1, 0], traj_ov_sel[-1, 1], size_ov_sel[-1, 0], size_ov_sel[-1, 1], 1.5,
                              hcolor_ov_sel, sub_num=sub_num)

        # Draw other vehicles ([id_cf]) ----------------------------------------------------------#
        if len(traj_ov_cf) > 0:
            hcolor_cf = get_rgb('Gold')
            self.draw_traj(traj_ov_cf[:, 0:2], 2.5, hcolor_cf, sub_num=sub_num)
            self.draw_pnt(traj_ov_cf[0, 0:2], 5, hcolor_cf, is_fill=True, sub_num=sub_num)
            # self.draw_target_vehicle_fill(traj_ov_cf[0, 0], traj_ov_cf[0, 1], 0.0, size_ov_cf[0, 0], size_ov_cf[0, 1],
            #                               get_color_blurred(hcolor_cf), sub_num=sub_num)
            self.draw_box(traj_ov_cf[0, 0], traj_ov_cf[0, 1], size_ov_cf[0, 0], size_ov_cf[0, 1], 3.0, hcolor_cf,
                          sub_num=sub_num)
            self.draw_box(traj_ov_cf[-1, 0], traj_ov_cf[-1, 1], size_ov_cf[-1, 0], size_ov_cf[-1, 1], 1.5, hcolor_cf,
                          sub_num=sub_num)

        # Draw ego-vehicle -----------------------------------------------------------------------#
        hcolor_ev = get_rgb('Dark Pastel Blue')
        hcolor_ev_blurred = get_color_blurred(hcolor_ev)
        self.draw_target_vehicle_fill(xinit_conv[0], xinit_conv[1], 0.0, rx, ry, hcolor_ev_blurred, sub_num=2)
        self.draw_traj(traj_computed[:, 0:2], 3.0, hcolor_ev, sub_num=sub_num)
        self.draw_pnt(traj_computed[0, 0:2], 5, hcolor_ev, is_fill=True, sub_num=sub_num)
        self.draw_box(traj_computed[0, 0], traj_computed[0, 1], rx, ry, 3.0, hcolor_ev, sub_num=sub_num)
        self.draw_box(traj_computed[-1, 0], traj_computed[-1, 1], rx, ry, 1.5, hcolor_ev, sub_num=sub_num)

        self.draw_pnt(xgoal_conv[0:2], 5, get_rgb('Red'), is_fill=False, sub_num=sub_num)

        # Draw text ------------------------------------------------------------------------------#
        fontsize_txt = 20
        hcolor_txt = get_rgb('White') if self.is_black else get_rgb('Black')
        self.draw_text(xinit_conv[0:2], 'ev', fontsize_txt, hcolor_txt, is_bold=1, sub_num=sub_num)

        if id_cf > 0:
            if len(traj_ov_cf) > 0:
                self.draw_text(traj_ov_cf[0, 0:2], 'cf', fontsize_txt, hcolor_txt, is_bold=1, sub_num=sub_num)

        txt_ov = ['lf', 'lr', 'rf', 'rr', 'cr']
        for nidx_ov in range(0, 5):
            traj_ov_sel = traj_ov[nidx_ov]
            if id_rest[nidx_ov] > 0 and len(traj_ov_sel) > 0:
                self.draw_text(traj_ov_sel[0, 0:2], txt_ov[nidx_ov], fontsize_txt, hcolor_txt, is_bold=1,
                               sub_num=sub_num)

    # -----------------------------------------------------------------------------------------------------------------#
    # DRAW-CONTROL ----------------------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    def draw_ctrl_basic(self, data_ev_cur, data_ov_cur, id_ev, id_near, id_rest, traj_ev, traj_hist_ev,
                        traj_ov_near, size_ov_near, v_arrow,
                        traj_array_ev=None, costtraj_array_ev=None, inv_idxtraj_array_ev=None, hcolor_map_ev="cool",
                        hcolor_reverse_ev=False,
                        predtraj_ov=None, size_ov_gt=None):
        # Color setting
        hcolor_fill_ev = get_rgb("Crayola")
        hcolor_fill_ovsel, hcolor_fill_rest = get_rgb("Salmon"), get_rgb("Dark Gray")
        hcolor_arrow_sel, hcolor_arrow_rest = get_rgb("White"), get_rgb("Dark Gray")
        hcolor_border_ev = get_rgb("Han Blue")
        hcolor_border_ovsel, hcolor_border_ovrest = get_rgb("Dark Pastel Red"), get_rgb("Dim Gray")

        hcolor_traj_ev, hcolor_traj_ov = get_rgb("Han Blue"), get_rgb("Indian Red")
        hcolor_border_traj_ev = get_rgb("Corn Flower Blue")
        hcolor_hist_traj_ev = get_rgb("Dark Turquoise")

        hcolor_predtraj_ov = get_rgb("Red")

        traj_array_ev = [] if traj_array_ev is None else traj_array_ev
        costtraj_array_ev = [] if costtraj_array_ev is None else costtraj_array_ev
        inv_idxtraj_array_ev = [] if inv_idxtraj_array_ev is None else inv_idxtraj_array_ev

        predtraj_ov = [] if predtraj_ov is None else predtraj_ov

        # Draw other-vehicle
        self.draw_vehicle_fill(data_ov_cur, id_near, hcolor_fill_ovsel, op=0.9)

        # Draw trajectories (other-vehicle)
        if len(traj_ov_near) > 0:
            self.draw_trajs(traj_ov_near, 2, hcolor_traj_ov)
            self.draw_vehicle_fill_trajs_cgad(traj_ov_near, size_ov_near, 2, hcolor_fill_ovsel, hcolor_fill_rest, op=0.75)
            self.draw_vehicle_border_trajs(traj_ov_near, size_ov_near, 2, 1.5, hcolor_traj_ov, op=0.8)

        if len(predtraj_ov) > 0:
            for nidx_n in range(0, len(id_near)):
                if id_near[nidx_n] != -1:
                    predtraj_ov_sel = predtraj_ov[nidx_n]
                    size_ov_gt_sel = size_ov_gt[nidx_n]
                    self.draw_vehicle_border_traj(predtraj_ov_sel, size_ov_gt_sel[0], size_ov_gt_sel[1], 2,
                                                  1.5, hcolor_predtraj_ov, op=1.0)
                    self.draw_traj(predtraj_ov_sel, 2.0, hcolor_predtraj_ov, op=1.0)

        # Draw traj-hist (ego-vehicle)
        if len(traj_hist_ev) > 0:
            self.draw_traj(traj_hist_ev, 2.2, hcolor_hist_traj_ev)

        # Draw arrows (other-vehicle)
        self.draw_vehicle_arrow(data_ov_cur, id_near, 20, hcolor_arrow_sel, op=1.0)
        self.draw_vehicle_arrow(data_ov_cur, id_rest, 20, hcolor_arrow_rest, op=1.0)

        # Draw trajectories (ego-vehicle)
        if len(traj_ev) > 0:
            self.draw_vehicle_fill_traj_cgad(traj_ev, data_ev_cur[6], data_ev_cur[5], 2, hcolor_fill_ev,
                                             hcolor_fill_rest, op=0.75)

        if len(traj_array_ev) > 0 and len(costtraj_array_ev) > 0:
            self.draw_trajs_w_cost(traj_array_ev, costtraj_array_ev, inv_idxtraj_array_ev, 0, 2.0,
                                   hcolor_map=hcolor_map_ev, op=0.8, hcolor_reverse=hcolor_reverse_ev)

        if len(traj_ev) > 0:
            self.draw_vehicle_border_traj(traj_ev, data_ev_cur[6], data_ev_cur[5], 2, 2.5, hcolor_border_traj_ev,
                                          op=1.0)
            self.draw_traj(traj_ev, 2.0, hcolor_traj_ev, op=1.0)
            # traj_sel_ev_ = interpolate_traj(traj_sel_ev, alpha=3)
            # self.draw_traj_dashed(traj_sel_ev_, 2.3, hcolor_traj_ev)

        # Draw ego-vehicle
        self.draw_target_vehicle_fill(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6], data_ev_cur[5],
                                      hcolor_fill_ev, op=0.9)
        self.draw_target_vehicle_arrow(data_ev_cur[1], data_ev_cur[2], data_ev_cur[3], data_ev_cur[6],
                                       data_ev_cur[5], data_ev_cur[4], v_arrow, hcolor_arrow_sel)

        # Draw borders
        self.draw_vehicle_border(data_ov_cur, id_near, 2.0, hcolor_border_ovsel, op=1)
        self.draw_vehicle_border(data_ov_cur, id_rest, 2.0, hcolor_border_ovrest, op=1)
        self.draw_vehicle_border(data_ev_cur, [id_ev], 3.0, hcolor_border_ev, op=1)

    # -----------------------------------------------------------------------------------------------------------------#
    # UPDATE-SCREEN (DISPLAY) -----------------------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------------------------------------------#
    def display(self):
        """ Display screen. """
        # Get data surface
        data_surface = self.bgra_surf_to_rgba_string()

        # Create pygame surface
        pygame_surface = pygame.image.frombuffer(data_surface, (self.screen_size[0], self.screen_size[1]), 'RGBA')

        # Show pygame surface
        self.pygame_screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()

        # Limit frame-rate (frames per second)
        self.clock.tick(self.clock_rate)
