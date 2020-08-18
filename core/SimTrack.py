from __future__ import print_function

import math
import numpy as np


class SimTrack(object):
    """
    TRACK
    1. NGSIM (I80 & US101)
        Reference: Vassili Alexiadis, James Colyar, John Halkias, Rob Hranac, and Gene McHale,
        "The Next Generation Simulation Program," Institute of Transportation Engineers, ITE Journal 74,
        no. 8 (2004): 22.
    2. highD (tracknum: 1~60)
        Reference: Robert Krajewski, Julian Bock, Laurent Kloeker and Lutz Eckstein,
        "The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of
        Highly Automated Driving Systems." in Proc. of the IEEE 21st International Conference on Intelligent
        Transportation Systems (ITSC), 2018
    """
    def __init__(self, track_name_in):
        self.track_name = track_name_in
        self.num_seg = []
        self.num_lane = []

        self.pnts_poly_track = []
        self.pnts_outer_border_track = []
        self.pnts_inner_border_track = []
        self.pnts_lr_border_track = []  # [0, :] start --> [end, :] end
        self.pnts_m_track = []  # [0, :] start --> [end, :] end

        self.pnt_mean, self.pnt_min, self.pnt_max = [], [], []

        self.is_circular = False

        # Lane type
        self.lane_type = []

        # Lane dir
        self.lane_dir = []

        # Parent & Child indexes
        self.idx_parent, self.idx_child = [], []

        # Goal indexes & points
        self.indexes_goal, self.pnts_goal = [], []

        self.th_lane_connected_lower = []  # Threshold for whether two lanes are connected (lower)
        self.th_lane_connected_upper = []  # Threshold for whether two lanes are connected (upper)

        # Points of interest
        self.p0, self.p1, self.p2, self.p3 = [], [], [], []

        # Screen size (width x height)
        self.screen_size_wide = np.array([700, 700], dtype=np.float32)
        self.screen_size_narrow = np.array([700, 700], dtype=np.float32)
        self.screen_alpha_wide, self.screen_alpha_narrow = 1, 1
        self.screen_sub_height_ratio = 1/4

        if ("US101" in self.track_name) or ("us101" in self.track_name):
            # print("Get track-info of \"US101\"")
            self.trackname = 'us101'
            self.load_us101()
        elif ("I80" in self.track_name) or ("i80" in self.track_name):
            # print('Get track-info of \"I80\"')
            self.trackname = 'i80'
            self.load_i80()
        elif ("highD" in self.track_name) or ("highd" in self.track_name):
            # print('Get track-info of \"highD\"')
            self.trackname = 'highd'
            self.load_highD()
        elif ("TOY" in self.track_name) or ("toy" in self.track_name):
            # print('Get track-info of \"toy\"')
            self.trackname = 'toy'
            if ("Circular" in self.track_name) or ("circular" in self.track_name):
                self.is_circular = True
            self.load_toy()
        else:
            # Skip
            print('Do nothing')

    def read_params_track(self, filename2read):
        data_read = np.load(filename2read, allow_pickle=True)

        num_seg, num_lane = data_read[()]["num_seg"], data_read[()]["num_lane"]
        pnts_poly_track = data_read[()]["pnts_poly_track"]
        pnts_outer_border_track = data_read[()]["pnts_outer_border_track"]
        pnts_inner_border_track = data_read[()]["pnts_inner_border_track"]
        pnts_lr_border_track = data_read[()]["pnts_lr_border_track"]
        pnts_m_track = data_read[()]["pnts_m_track"]

        pnt_mean, pnt_min, pnt_max = data_read[()]["pnt_mean"], data_read[()]["pnt_min"], data_read[()]["pnt_max"]
        lane_type, lane_dir = data_read[()]["lane_type"], data_read[()]["lane_dir"]
        idx_parent, idx_child = data_read[()]["idx_parent"], data_read[()]["idx_child"]
        indexes_goal, pnts_goal = data_read[()]["indexes_goal"], data_read[()]["pnts_goal"]
        th_lane_connected_lower, th_lane_connected_upper = data_read[()]["th_lane_connected_lower"], \
                                                           data_read[()]["th_lane_connected_upper"]

        if "i80" in self.track_name or "I80" in self.track_name:
            p0, p1, p2 = data_read[()]["p0"], data_read[()]["p1"], data_read[()]["p2"]
            self.p0, self.p1, self.p2 = p0, p1, p2
        elif "us101" in self.track_name or "US101" in self.track_name:
            p0, p1, p2 = data_read[()]["p0"], data_read[()]["p1"], data_read[()]["p2"]
            self.p0, self.p1, self.p2 = p0, p1, p2
        else:
            pass

        self.num_seg, self.num_lane = num_seg, num_lane
        self.pnts_poly_track = pnts_poly_track
        self.pnts_outer_border_track, self.pnts_inner_border_track = pnts_outer_border_track, pnts_inner_border_track
        self.pnts_lr_border_track, self.pnts_m_track = pnts_lr_border_track, pnts_m_track

        self.pnt_mean, self.pnt_min, self.pnt_max = pnt_mean, pnt_min, pnt_max
        self.lane_type, self.lane_dir = lane_type, lane_dir
        self.idx_parent, self.idx_child = idx_parent, idx_child
        self.indexes_goal, self.pnts_goal = indexes_goal, pnts_goal
        self.th_lane_connected_lower, self.th_lane_connected_upper = th_lane_connected_lower, th_lane_connected_upper

    def load_us101(self):
        filename2read = "./data_track/params_track_ngsim_us101.npy"

        self.read_params_track(filename2read)

        # Screen size (width x height)
        ratio_height2width = (self.pnt_max[1] - self.pnt_min[1]) / (self.pnt_max[0] - self.pnt_min[0])
        width_screen = 2400.0
        height_screen = math.ceil(width_screen * ratio_height2width * 1.05)
        self.screen_size_wide = np.array([width_screen, height_screen], dtype=np.int32)
        ratio_height2width_narrow = 6.25 / 10.0
        width_screen_narrow = 1200.0
        self.screen_size_narrow = np.array([width_screen_narrow, width_screen_narrow * ratio_height2width_narrow], dtype=np.int32)
        self.screen_alpha_wide, self.screen_alpha_narrow = width_screen / (self.pnt_max[0] - self.pnt_min[0]), 11.0
        self.screen_sub_height_ratio = 1 / 12.5

    def load_i80(self):
        filename2read = "./data_track/params_track_ngsim_i80.npy"

        self.read_params_track(filename2read)

        # Screen size (width x height)
        ratio_height2width = (self.pnt_max[1] - self.pnt_min[1]) / (self.pnt_max[0] - self.pnt_min[0])
        width_screen = 2400.0
        height_screen = math.ceil(width_screen * ratio_height2width * 1.05)
        self.screen_size_wide = np.array([width_screen, height_screen], dtype=np.int32)
        ratio_height2width_narrow = 6.25 / 10.0
        width_screen_narrow = 1200.0
        self.screen_size_narrow = np.array([width_screen_narrow, width_screen_narrow * ratio_height2width_narrow], dtype=np.int32)
        self.screen_alpha_wide, self.screen_alpha_narrow = width_screen / (self.pnt_max[0] - self.pnt_min[0]), 11.0
        self.screen_sub_height_ratio = 1 / 12.5

    def load_highD(self):
        text_split = self.track_name.split("_")
        track_num = text_split[-1]

        filename2read = "./data_track/params_track_highD_{:s}.npy".format(track_num)

        self.read_params_track(filename2read)

        # Screen size (width x height)
        ratio_height2width = (self.pnt_max[1] - self.pnt_min[1]) / (self.pnt_max[0] - self.pnt_min[0])
        width_screen = 2400.0
        height_screen = int(width_screen * ratio_height2width * 1.0)
        self.screen_size_wide = np.array([width_screen, height_screen], dtype=np.int32)
        ratio_height2width_narrow = 6.25 / 10.0
        width_screen_narrow = 1200.0
        self.screen_size_narrow = np.array([width_screen_narrow, width_screen_narrow * ratio_height2width_narrow], dtype=np.int32)
        self.screen_alpha_wide, self.screen_alpha_narrow = height_screen / (self.pnt_max[1] - self.pnt_min[1]), 11.0
        self.screen_sub_height_ratio = 1.0 / 12.0

    def load_toy(self):
        text_split = self.track_name.split("_")
        track_type = text_split[-2]
        track_num = text_split[-1]

        filename2read = "./data_track/params_track_toy_{:s}_{:s}.npy".format(track_type, track_num)

        self.read_params_track(filename2read)

        # Screen size (width x height)
        ratio_width2height = (self.pnt_max[1] - self.pnt_min[1]) / (self.pnt_max[0] - self.pnt_min[0])
        # ratio_height2width = (self.pnt_max[0] - self.pnt_min[0]) / (self.pnt_max[1] - self.pnt_min[1])
        width_screen = 1000.0
        height_screen = int(width_screen * ratio_width2height * 1.0)
        self.screen_size_wide = np.array([width_screen, height_screen], dtype=np.int32)
        self.screen_size_narrow = np.array([900, 900], dtype=np.int32)
        self.screen_alpha_wide, self.screen_alpha_narrow = height_screen / (self.pnt_max[1] - self.pnt_min[1]), 15
        self.screen_sub_height_ratio = 1 / 5


if __name__ == '__main__':
    trackname = "toy_circular_2"  # "us101", "i80", "highd_1", "toy_circular_2"

    sim_track = SimTrack(trackname)
    num_lane_max = max(sim_track.num_lane)

    pnts_m = sim_track.pnts_m_track[0][0]

    # print(sim_track.lane_type[0][0] == "Straight")
    print(sim_track.num_lane[0])

    # PLOT TRACK
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(1, 1)
    cmap = cm.get_cmap("rainbow")
    for nidx_seg in range(0, len(sim_track.pnts_poly_track)):
        seg_pnts_poly_track = sim_track.pnts_poly_track[nidx_seg]
        pnts_m_track = sim_track.pnts_m_track[nidx_seg]
        pnts_lr_track = sim_track.pnts_lr_border_track[nidx_seg]

        for nidx_lane in range(0, len(seg_pnts_poly_track)):
            lane_pnts_poly_track = seg_pnts_poly_track[nidx_lane]
            pnts_m_lane = pnts_m_track[nidx_lane]
            pnts_lr_lane = pnts_lr_track[nidx_lane]

            cmap_sel = cmap(nidx_lane / num_lane_max)
            ax.plot(lane_pnts_poly_track[:, 0], lane_pnts_poly_track[:, 1], color=cmap_sel)
            # ax.plot(pnts_m_lane[:, 0], pnts_m_lane[:, 1], 'b-')
            ax.plot(pnts_m_lane[0, 0], pnts_m_lane[0, 1], 'bo')

            if nidx_lane == 0:
                ax.plot(pnts_lr_lane[0][:, 0], pnts_lr_lane[0][:, 1], 'r.')
                ax.plot(pnts_lr_lane[1][:, 0], pnts_lr_lane[1][:, 1], 'g.')

                pnt_l_x_tmp = [pnts_lr_lane[0][0, 0], pnts_lr_lane[0][-1, 0]]
                pnt_l_y_tmp = [pnts_lr_lane[0][0, 1], pnts_lr_lane[0][-1, 1]]
                ax.plot(pnt_l_x_tmp, pnt_l_y_tmp, 'r-')

    ax.plot(sim_track.pnt_mean[0], sim_track.pnt_mean[1], 'rx')
    ax.plot(sim_track.pnt_min[0], sim_track.pnt_min[1], 'rs')
    ax.plot(sim_track.pnt_max[0], sim_track.pnt_max[1], 'ro')

    if len(sim_track.pnts_goal) > 0:
        ax.plot(sim_track.pnts_goal[:, 0], sim_track.pnts_goal[:, 1], 'y*')
    ax.axis("equal")
    plt.show()
