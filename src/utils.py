# UTILITY-FUNCTIONS (BASIC)

import numpy as np
import math

from sklearn import preprocessing
from matplotlib import path

__all__ = ["is_two_lists_equal", "make_numpy_array", "get_two_united_set", "set_traj_in_range", "set_vector_in_range",  # Utils
           "angle_handle", "get_diff_angles", "norm", "get_l2_loss", "inpolygon",
           "normalize_data", "normalize_data_wrt_mean_scale", "recover_from_normalized_data",  # Normalize
           "get_rotated_pnts_rt", "get_rotated_pnts_tr", "get_box_pnts", "get_box_pnts_precise", "get_m_pnts",  # Get points
           "get_intp_pnts", "get_intp_pnts_wrt_dist", "get_pnts_carshape", "get_pnts_arrow", "get_dist_point2line",
           "get_closest_pnt", "get_closest_pnt_intp", "get_sparse_pnts_2d_grid",
           "interpolate_w_ratio", "interpolate_data", "interpolate_traj",  # Interpolate
           "apply_gaussian_kde_2d_naive"]


def is_two_lists_equal(x1, x2):
    """ Checks if two lists are identical.
    :param x1: list1
    :param x2: list2 """

    # Sorting both the lists
    x1.sort()
    x2.sort()

    # Using == to check if lists are equal
    if x1 == x2:
        # print("The lists are identical")
        is_equal = True
    else:
        print("The lists are not identical")
        is_equal = False

    return is_equal


def make_numpy_array(x, keep_1dim=False):
    """ Makes numpy-array.
    :param x: (list) input data """
    is_numeric_type = isinstance(x, float) or isinstance(x, int) or isinstance(x, np.int8) or isinstance(x, np.int16) \
                      or isinstance(x, np.int32) or isinstance(x, np.float16) or isinstance(x, np.float32) or \
                      isinstance(x, np.float64)

    if x is None:
        x = []
    elif is_numeric_type:
        x = np.array([x])
    elif len(x) > 0:
        if ~isinstance(x, np.ndarray):
            x = np.array(x)
        shape_x = x.shape
        if keep_1dim:
            x = x.reshape(-1)
        else:
            if len(shape_x) == 1:
                x = np.reshape(x, (1, -1))
    else:
        x = []

    return x


def get_two_united_set(u0_set, u1_set):
    """ Gets united two set.
    :param u0_set: set 1 (dim = N)
    :param u1_set: set 2 (dim = N)
    :return: u_set
    """
    # u0_set, u1_set: (ndarray) set of controls for each dimension (dim = N)

    u0_set = make_numpy_array(u0_set, keep_1dim=True)
    u1_set = make_numpy_array(u1_set, keep_1dim=True)

    len_u0 = u0_set.shape[0]
    len_u1 = u1_set.shape[0]

    u_set = np.zeros((len_u0 * len_u1, 2), dtype=np.float32)

    u0_range_ext = np.repeat(u0_set, len_u1)
    u1_range_ext = np.tile(u1_set, len_u0)

    u_set[:, 0] = u0_range_ext
    u_set[:, 1] = u1_range_ext

    return u_set


def set_traj_in_range(traj, pnt_min, pnt_max):
    """ Sets trajectory in range.
    :param traj: input trajectory (ndarray, dim = N x 2)
    :param pnt_min: min point (ndarray, dim = 2)
    :param pnt_max: max point (ndarray, dim = 2)
    """
    traj = make_numpy_array(traj, keep_1dim=False)
    pnt_min = make_numpy_array(pnt_min, keep_1dim=True)
    pnt_max = make_numpy_array(pnt_max, keep_1dim=True)

    idx_tmp0 = np.where(traj[:, 0] <= pnt_max[0])
    idx_tmp0 = idx_tmp0[0]
    idx_tmp1 = np.where(traj[:, 0] >= pnt_min[0])
    idx_tmp1 = idx_tmp1[0]
    idx_tmp2 = np.where(traj[:, 1] <= pnt_max[1])
    idx_tmp2 = idx_tmp2[0]
    idx_tmp3 = np.where(traj[:, 1] >= pnt_min[1])
    idx_tmp3 = idx_tmp3[0]

    idx_tmp01 = np.intersect1d(idx_tmp0, idx_tmp1)
    idx_tmp23 = np.intersect1d(idx_tmp2, idx_tmp3)
    idx_tmp0123 = np.intersect1d(idx_tmp01, idx_tmp23)

    traj_out = traj[idx_tmp0123, :]

    return traj_out


def set_vector_in_range(vec_in, vec_min, vec_max):
    """ Sets vector in range.
    :param vec_in: vector-in (dim = N)
    :param vec_min: min-vector (dim = N)
    :param vec_max: max-vector (dim = N)
    """
    vec_in = make_numpy_array(vec_in, keep_1dim=True)
    vec_min = make_numpy_array(vec_min, keep_1dim=True)
    vec_max = make_numpy_array(vec_max, keep_1dim=True)

    vec_new = np.maximum(vec_in, vec_min)
    vec_new = np.minimum(vec_new, vec_max)

    return vec_new


def angle_handle(q, mode=0):
    """ Adjusts angle.
    :param q: angle
    :param mode: 0-> -pi ~ +pi, 1-> 0 ~ +2*pi
    :return: q_new
    """
    q = make_numpy_array(q, keep_1dim=True)
    q_new = np.copy(q)

    if mode == 0:
        q_low, q_high = -math.pi, +math.pi
    else:
        q_low, q_high = 0.0, 2.0 * math.pi

    idx_found_upper = np.where(q > q_high)
    idx_found_upper = idx_found_upper[0]

    idx_found_lower = np.where(q <= q_low)
    idx_found_lower = idx_found_lower[0]

    if len(idx_found_upper) > 0:
        q_new[idx_found_upper] = q[idx_found_upper] - 2 * math.pi
    if len(idx_found_lower) > 0:
        q_new[idx_found_lower] = q[idx_found_lower] + 2 * math.pi

    return q_new


def get_diff_angles(q1, q2):
    """ Gets difference between two angles (q1 - q2).
    :param q1: angle1
    :param q2: angle2 """
    if abs(q1 - q2) < abs(q1 + q2):
        diff_q = q1 - q2
    else:
        diff_q = q1 + q2

    return angle_handle(diff_q)


def norm(vec):
    """ Gets l2-norm.
    :param vec: vector (dim = N)
    :return: len_vec
    """
    vec = make_numpy_array(vec, keep_1dim=True)
    vec_ = vec * vec
    len_vec = np.sqrt(np.sum(vec_))

    return len_vec


def get_l2_loss(x1, x2):
    """ Gets l2-loss.
    :param x1: data (dim = N x m)
    :param x2: data (dim = N x m)
    """
    x1 = make_numpy_array(x1, keep_1dim=False)
    x2 = make_numpy_array(x2, keep_1dim=False)

    diff_tmp = x1 - x2
    l2_array = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    l2_out = np.sum(l2_array)
    return l2_array, l2_out


def inpolygon(xq, yq, xv, yv):
    """ Checks whether point is in the polygon.
    :param xq: points query x (dim = Nq)
    :param yq: points query y (dim = Nq)
    :param xv: points vertex x (dim = Nv)
    :param yv: points vertex y (dim = Nv)
    :return: whether vertex points is in query.
    """
    xq = make_numpy_array(xq, keep_1dim=True)
    yq = make_numpy_array(yq, keep_1dim=True)
    xv = make_numpy_array(xv, keep_1dim=True)
    yv = make_numpy_array(yv, keep_1dim=True)

    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(-1)


def normalize_data(x):
    """ Normalizes data: find mean, std for normalization.
    :param x: (list or ndarray) input data """

    if len(x) == 0:
        x_standard = []
        x_mean = []
        x_scale = []
    else:
        x = make_numpy_array(x, keep_1dim=False)

        x_standard = preprocessing.StandardScaler().fit(x)

        x_scaled = x_standard.transform(x)
        x_mean = x_standard.mean_
        x_scale = x_standard.scale_

    return x_scaled, x_mean, x_scale


def normalize_data_wrt_mean_scale(x, x_mean, x_scale):
    """ Normalizes data w.r.t. mean & scale.
    x, x_mean, x_scale: (list or ndarray) input data, mean-data, scale-data """
    if len(x) == 0 or len(x_mean) == 0 or len(x_scale) == 0:
        x = []
    else:
        x = make_numpy_array(x, keep_1dim=False)

        _n = x.shape[0]
        x = x
        x = x - np.tile(x_mean, (_n, 1))
        x = x / np.tile(x_scale, (_n, 1))
    return x


def recover_from_normalized_data(x, x_mean, x_scale):
    """ Recovers from normalized data.
     x, x_mean, x_scale: (list or ndarray) input data, mean-data, scale-data """
    if len(x) == 0 or len(x_mean) == 0 or len(x_scale) == 0:
        x = []
    else:
        x = make_numpy_array(x, keep_1dim=False)
        x_mean = make_numpy_array(x_mean, keep_1dim=False)
        x_scale = make_numpy_array(x_scale, keep_1dim=False)

        _n = x.shape[0]
        x = x
        x = x * np.tile(x_scale, (_n, 1))
        x = x + np.tile(x_mean, (_n, 1))
    return x


def get_rotated_pnts_rt(pnts_in, cp, theta):
    """ Rotates points (Rotation --> Transition).
    :param pnts_in: points (dim = N x 2)
    :param cp: center-point (dim = 2)
    :param theta: rotation-angle (float)
    :return: points
    """
    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)
    cp = make_numpy_array(cp, keep_1dim=True)

    num_pnts = pnts_in.shape[0]

    R = np.array([[+math.cos(theta), +math.sin(theta)], [-math.sin(theta), +math.cos(theta)]], dtype=np.float32)
    cp_tmp = np.reshape(cp, (1, 2))
    cp_tmp = np.tile(cp_tmp, (num_pnts, 1))

    pnts_out = np.matmul(pnts_in, R) + cp_tmp

    return pnts_out


def get_rotated_pnts_tr(pnts_in, cp, theta):
    """ Rotates points (Transition --> Rotation).
    :param pnts_in: points (dim = N x 2)
    :param cp: center-point (dim = 2)
    :param theta: rotation-angle (float)
    :return: points
    """
    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)
    cp = make_numpy_array(cp, keep_1dim=True)

    num_pnts = pnts_in.shape[0]

    R = np.array([[+math.cos(theta), +math.sin(theta)], [-math.sin(theta), +math.cos(theta)]], dtype=np.float32)
    cp_tmp = np.reshape(cp, (1, 2))
    cp_tmp = np.tile(cp_tmp, (num_pnts, 1))

    pnts_out = np.matmul(pnts_in + cp_tmp, R)

    return pnts_out


def get_box_pnts(x, y, theta, length, width):
    """ Returns points of box-shape.
    :param x: position-x
    :param y: position-y
    :param theta: heading
    :param length: length
    :param width: width
    :return: points-box
    """
    rx, ry = length / 2.0, width / 2.0
    p0, p1, p2, p3 = [-rx, -ry], [+rx, -ry], [+rx, +ry], [-rx, +ry]

    pnts_box_ = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]], dtype=np.float32)
    pnts_box = get_rotated_pnts_rt(pnts_box_, [x, y], theta)

    return pnts_box


def get_box_pnts_precise(x, y, theta, length, width, nx=10, ny=10):
    """ Returns points of box-shape (precise).
    :param x: position-x
    :param y: position-y
    :param theta: heading
    :param length: length
    :param width: width
    :param nx: number of points (x)
    :param ny: number of points (y)
    :return: points-box
    """
    rx, ry = length / 2.0, width / 2.0

    l0_x = np.linspace(-rx, +rx, num=nx, dtype=np.float32)
    l0_y = -ry*np.ones((nx,), dtype=np.float32)
    l1_x = +rx*np.ones((ny,), dtype=np.float32)
    l1_y = np.linspace(-ry, +ry, num=ny, dtype=np.float32)
    l2_x = np.linspace(+rx, -rx, num=nx, dtype=np.float32)
    l2_y = +ry * np.ones((nx,), dtype=np.float32)
    l3_x = -rx * np.ones((ny,), dtype=np.float32)
    l3_y = np.linspace(-ry, +ry, num=ny, dtype=np.float32)

    pnts_x = np.concatenate((l0_x, l1_x, l2_x, l3_x), axis=0)
    pnts_y = np.concatenate((l0_y, l1_y, l2_y, l3_y), axis=0)
    pnts_x = pnts_x.reshape((-1, 1))
    pnts_y = pnts_y.reshape((-1, 1))

    pnts_box_ = np.concatenate((pnts_x, pnts_y), axis=1)
    pnts_box = get_rotated_pnts_rt(pnts_box_, [x, y], theta)

    return pnts_box


def get_m_pnts(x, y, theta, length, nx=10):
    """ Returns middle points.
    :param x: position-x
    :param y: position-y
    :param theta: heading
    :param length: length
    :param nx: number of points
    :return: m-points
    """
    rx = length / 2.0

    pnts_x = np.linspace(-rx, +rx, num=nx, dtype=np.float32)
    pnts_y = np.zeros((nx, ), dtype=np.float32)
    pnts_x = pnts_x.reshape((-1, 1))
    pnts_y = pnts_y.reshape((-1, 1))
    pnts_m_ = np.concatenate((pnts_x, pnts_y), axis=1)
    pnts_m = get_rotated_pnts_rt(pnts_m_, [x, y], theta)

    return pnts_m


def get_intp_pnts(pnts_in, num_intp):
    """ Gets interpolate points
    :param pnts_in: points (dim = N x 2)
    :param num_intp: interpolation number
    :return: intp-points
    """
    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)

    pnt_x_tmp = pnts_in[:, 0].reshape(-1)
    pnt_y_tmp = pnts_in[:, 1].reshape(-1)
    pnt_t_tmp = np.arange(0, pnt_x_tmp.shape[0])

    t_range = np.linspace(min(pnt_t_tmp), max(pnt_t_tmp), num=num_intp)

    x_intp = np.interp(t_range, pnt_t_tmp, pnt_x_tmp)
    y_intp = np.interp(t_range, pnt_t_tmp, pnt_y_tmp)

    pnts_intp = np.zeros((num_intp, 2), dtype=np.float32)
    pnts_intp[:, 0] = x_intp
    pnts_intp[:, 1] = y_intp

    return pnts_intp


def get_intp_pnts_wrt_dist(pnts_in, num_intp, dist_intv):
    """ Gets interpolate points (w.r.t. dist).
    :param pnts_in: points (dim = N x 2)
    :param dist_intv: distance interval
    :return: intp-points
    """
    pnts_in = make_numpy_array(pnts_in, keep_1dim=False)

    pnts_intp = get_intp_pnts(pnts_in, num_intp)
    len_pnts_intp = pnts_intp.shape[0]
    diff_tmp = pnts_intp[np.arange(0, len_pnts_intp - 1), 0:2] - pnts_intp[np.arange(1, len_pnts_intp), 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))

    dist_sum, cnt_intv = 0, 0
    idx_intv = np.zeros((20000, ), dtype=np.int32)
    for nidx_d in range(0, dist_tmp.shape[0]):
        dist_sum = dist_sum + dist_tmp[nidx_d]
        if dist_sum >= (cnt_intv + 1) * dist_intv:
            cnt_intv = cnt_intv + 1
            idx_intv[cnt_intv] = nidx_d

    idx_intv = idx_intv[0:cnt_intv + 1]

    if ~np.isin((pnts_intp.shape[0] - 1), idx_intv):
        idx_intv[-1] = pnts_intp.shape[0] - 1

    pnts_intv = pnts_intp[idx_intv, :]

    return pnts_intv


def get_pnts_carshape(x, y, theta, length, width):
    """ Returns points of car-shape.
    :param x: position-x
    :param y: position-y
    :param theta: heading
    :param length: length
    :param width: width
    """
    rx, ry = length / 2.0, width / 2.0
    lx = rx * 1.5
    th = math.atan2(ry, math.sqrt(lx * lx - ry * ry))

    p0 = np.array([rx - lx, 0.0], dtype=np.float64)
    p1 = np.array([-rx, +ry], dtype=np.float64)
    p2 = np.array([-rx, -ry], dtype=np.float64)
    # p3 = np.array([lx*math.cos(th) - lx + rx, -ry], dtype=np.float64)
    # p4 = np.array([lx*math.cos(th) - lx + rx, +ry], dtype=np.float64)

    # pnts curve connecting p3 to p4
    num_pnts_curve = 9
    th_curve = np.linspace(-th, +th, num_pnts_curve)
    pnts_curve = np.zeros((num_pnts_curve, 2), dtype=np.float64)
    pnts_curve[:, 0] = lx*np.cos(th_curve) + p0[0]
    pnts_curve[:, 1] = lx*np.sin(th_curve)

    # pnts (raw)
    pnts_vehicle = np.zeros((num_pnts_curve + 2, 2), dtype=np.float64)
    pnts_vehicle[0, :] = p1
    pnts_vehicle[1, :] = p2
    pnts_vehicle[2:, :] = pnts_curve

    # pnts (rotated)
    pnt_cp = np.array([x, y], dtype=np.float64)
    pnts_vehicle_r = get_rotated_pnts_rt(pnts_vehicle, pnt_cp, theta)

    return pnts_vehicle_r


def get_pnts_arrow(x, y, theta, ax, ay, bx, by):
    """ Returns points of arrow.
    :param x: position-x
    :param y: position-y
    :param theta: heading
    :param ax: arrow parameter
    :param ay: arrow parameter
    :param bx: arrow parameter
    :param by: arrow parameter
    """
    # x, y, theta: (float) position-x, position-y, heading
    # ax, ay, bx, by: (float) arrow parameters

    # bx < ax
    pnts_arrow = np.zeros((7, 2), dtype=np.float32)
    pnts_arrow[0, :] = [-ax, +ay]
    pnts_arrow[1, :] = [ax - bx, +ay]
    pnts_arrow[2, :] = [ax - bx, ay + by]
    pnts_arrow[3, :] = [ax, 0]
    pnts_arrow[4, :] = [ax - bx, -(ay + by)]
    pnts_arrow[5, :] = [ax - bx, -ay]
    pnts_arrow[6, :] = [-ax, -ay]

    # pnts (rotated)
    pnt_cp = np.array([x, y], dtype=np.float32)
    pnts_vehicle_r = get_rotated_pnts_rt(pnts_arrow, pnt_cp, theta)

    return pnts_vehicle_r


def get_dist_point2line(pnt_i, pnt_a, pnt_b):
    """ Gets distance from point to line.
    :param pnt_i: point (dim = 2)
    :param pnt_a: line-point (dim = 2)
    :param pnt_b: line-point (dim = 2)
    :return: mindist, minpnt
    """

    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnt_a = make_numpy_array(pnt_a, keep_1dim=True)
    pnt_b = make_numpy_array(pnt_b, keep_1dim=True)

    diff_ab = pnt_a[0:2] - pnt_b[0:2]
    dist_ab = np.sqrt(diff_ab[0]*diff_ab[0] + diff_ab[1]*diff_ab[1])
    if dist_ab == 0:
        # PNT_A == PNT_B
        diff_ia = pnt_i[0:2] - pnt_a[0:2]
        mindist = norm(diff_ia[0:2])
        minpnt = pnt_a
    else:
        # OTHERWISE
        vec_a2b = pnt_b[0:2] - pnt_a[0:2]
        vec_b2a = pnt_a[0:2] - pnt_b[0:2]
        vec_a2i = pnt_i[0:2] - pnt_a[0:2]
        vec_b2i = pnt_i[0:2] - pnt_b[0:2]

        dot_tmp1 = vec_a2i[0]*vec_a2b[0] + vec_a2i[1]*vec_a2b[1]
        dot_tmp2 = vec_b2i[0]*vec_b2a[0] + vec_b2i[1]*vec_b2a[1]
        if dot_tmp1 < 0:
            minpnt = pnt_a
        elif dot_tmp2 < 0:
            minpnt = pnt_b
        else:
            len_a2b = norm(vec_a2b[0:2])
            minpnt = pnt_a + dot_tmp1 * vec_a2b / len_a2b / len_a2b

        diff_tmp = minpnt[0:2] - pnt_i[0:2]
        mindist = np.sqrt(diff_tmp[0]*diff_tmp[0] + diff_tmp[1]*diff_tmp[1])

    return mindist, minpnt


def get_closest_pnt(pnt_i, pnts_line):
    """ Gets the closest point.
    :param pnt_i: point (x, y)
    :param pnts_line: line-points (dim = N x 2)
    :return: pnt, mindist
    """
    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnts_line = make_numpy_array(pnts_line, keep_1dim=False)

    m = pnts_line.shape[0]
    mindist = 1e8

    pnt = []
    for nidx_i in range(0, m - 1):
        pnt_a = pnts_line[nidx_i, :]
        pnt_b = pnts_line[nidx_i + 1, :]

        [dist, cpnt] = get_dist_point2line(pnt_i, pnt_a, pnt_b)
        if dist < mindist:
            mindist = dist
            pnt = cpnt

    return pnt, mindist


def get_closest_pnt_intp(pnt_i, pnts_line, num_intp=100):
    """ Gets the closest point using interpolate.
    :param pnt_i: point (x, y, dim=2)
    :param pnts_line: line-points (dim = N x 2)
    :param num_intp: number of interpolation
    :return: pnt_out, mindist_out
    """
    pnt_i = make_numpy_array(pnt_i, keep_1dim=True)
    pnts_line = make_numpy_array(pnts_line, keep_1dim=False)

    len_pnts_line = pnts_line.shape[0]
    pnt_i_r = np.reshape(pnt_i[0:2], (1, 2))
    diff_tmp = np.tile(pnt_i_r, (len_pnts_line, 1)) - pnts_line[:, 0:2]
    dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
    idx_min = np.argmin(dist_tmp, axis=0)
    pnt_cur = pnts_line[idx_min, 0:2]
    pnt_cur = np.reshape(pnt_cur, (1, -1))

    if idx_min == 0:
        pnt_next = pnts_line[idx_min + 1, 0:2]
        pnt_next = np.reshape(pnt_next, (1, -1))
        pnts_tmp = np.concatenate((pnt_cur, pnt_next), axis=0)
        pnts_tmp_intp = get_intp_pnts(pnts_tmp, num_intp)
    elif idx_min == (len_pnts_line - 1):
        pnt_prev = pnts_line[idx_min - 1, 0:2]
        pnt_prev = np.reshape(pnt_prev, (1, -1))
        pnts_tmp = np.concatenate((pnt_prev, pnt_cur), axis=0)
        pnts_tmp_intp = get_intp_pnts(pnts_tmp, num_intp)
    else:
        pnt_prev = pnts_line[idx_min - 1, 0:2]
        pnt_prev = np.reshape(pnt_prev, (1, -1))
        pnts_tmp1 = np.concatenate((pnt_prev, pnt_cur), axis=0)
        pnts_tmp_intp1 = get_intp_pnts(pnts_tmp1, num_intp)
        pnt_next = pnts_line[idx_min + 1, 0:2]
        pnt_next = np.reshape(pnt_next, (1, -1))
        pnts_tmp2 = np.concatenate((pnt_cur, pnt_next), axis=0)
        pnts_tmp_intp2 = get_intp_pnts(pnts_tmp2, num_intp)
        pnts_tmp_intp = np.concatenate((pnts_tmp_intp1, pnts_tmp_intp2), axis=0)

    diff_tmp_new = np.tile(pnt_i_r, (pnts_tmp_intp.shape[0], 1)) - pnts_tmp_intp[:, 0:2]
    dist_tmp_new = np.sqrt(np.sum(diff_tmp_new * diff_tmp_new, axis=1))
    idx_cur_new = np.argmin(dist_tmp_new, axis=0)

    pnt_out = pnts_tmp_intp[idx_cur_new, :]
    mindist_out = dist_tmp_new[idx_cur_new]

    return pnt_out, mindist_out


def get_sparse_pnts_2d_grid(pnts, res, n_max=4, min_nx=5, min_ny=5):
    """ Gets sparse (2D) points w.r.t. grid.
    :param pnts: data (dim = N x 2)
    :param res: grid resolution (dim = 2)
    :param n_max: maximum number point in each grid (int)
    :param min_nx: minimum number grid (x) (int)
    :param min_ny: minimum number grid (y) (int)
    """
    pnts = make_numpy_array(pnts, keep_1dim=False)
    pnts = pnts[:, 0:2]
    len_in = pnts.shape[0]

    # Set grid points
    xmin, xmax = np.amin(pnts[:, 0]), np.amax(pnts[:, 0])
    ymin, ymax = np.amin(pnts[:, 1]), np.amax(pnts[:, 1])

    xrange, yrange = (xmax - xmin), (ymax - ymin)
    num_x, num_y = int(xrange / res[0]), int(yrange / res[1])
    num_x, num_y = max(num_x, min_nx), max(num_y, min_ny)

    x_grid_, y_grid_ = np.linspace(xmin, xmax, num_x), np.linspace(ymin, ymax, num_y)
    x_grid, y_grid = np.meshgrid(x_grid_, y_grid_)
    x_grid, y_grid = x_grid.ravel(), y_grid.ravel()

    pnts_grid = get_two_united_set(x_grid, y_grid)
    len_grid = pnts_grid.shape[0]

    # Check points w.r.t. grid points
    idx_grid = np.zeros((len_in,), dtype=np.int32)
    for nidx_d in range(0, len_in):
        pnt_tmp = pnts[nidx_d, :]
        pnt_tmp = np.reshape(pnt_tmp, (1, -1))
        pnt_tmp_ext = np.tile(pnt_tmp, (len_grid, 1))
        diff_tmp = pnt_tmp_ext - pnts_grid
        dist_tmp = np.sqrt(np.sum(diff_tmp * diff_tmp, axis=1))
        dist_tmp = np.reshape(dist_tmp, -1)
        idx_min_tmp = np.argmin(dist_tmp)

        idx_grid[nidx_d] = idx_min_tmp

    # Set output points
    pnt_out = np.zeros((len_in, 2), dtype=np.float32)
    cnt_out = 0
    for nidx_m in range(0, len_grid):
        idx_found_tmp = np.where(idx_grid == nidx_m)
        idx_found_tmp = idx_found_tmp[0]

        if len(idx_found_tmp) > n_max:
            idx_rand_tmp_ = np.random.permutation(len(idx_found_tmp))
            idx_rand_tmp = idx_rand_tmp_[0:n_max]
            idx_found_tmp = idx_found_tmp[idx_rand_tmp]

        idx_update_tmp = np.arange(cnt_out, cnt_out + len(idx_found_tmp))
        pnt_out[idx_update_tmp, :] = pnts[idx_found_tmp, :]
        cnt_out += len(idx_found_tmp)
    pnt_out = pnt_out[0:cnt_out, :]

    return pnt_out, pnts_grid


def interpolate_w_ratio(p1, p2, r_a, r_b):
    """ Interpolates two points w.r.t ratio.
    :param p1: point 1 (dim = 2)
    :param p2: point 2 (dim = 2)
    :param r_a: ratio 1
    :param r_b: ratio 2
    :return: p3 (interpolated points)
    """
    p1 = make_numpy_array(p1, keep_1dim=True)
    p2 = make_numpy_array(p2, keep_1dim=True)

    x_tmp = (r_b * p1[0] + r_a * p2[0]) / (r_a + r_b)
    y_tmp = (r_b * p1[1] + r_a * p2[1]) / (r_a + r_b)
    p3 = np.array([x_tmp, y_tmp], dtype=np.float32)
    return p3


def interpolate_data(data_in, data_type=0, alpha=2):
    """ Interpolates data.
    :param data_in: input data
    :param data_type: 0: len_new = len * alpha, 1: len_new = (len - 1) * alpha + 1
    :param alpha: interpolation multiplier
    """
    if len(data_in) > 0:
        data_in = make_numpy_array(data_in, keep_1dim=False)
        len_data_in = data_in.shape[0]
        dim_data_in = data_in.shape[1]
        t_in = np.arange(0, len_data_in)

        if data_type == 0:
            len_data_out = len_data_in * alpha
            t_out = np.linspace(start=t_in[0], stop=t_in[-1], num=len_data_out)
        else:
            len_data_out = 1 + (len_data_in - 1) * alpha
            t_out = np.linspace(start=t_in[0], stop=t_in[-1], num=len_data_out)

        data_out = np.zeros((len_data_out, dim_data_in), dtype=np.float32)
        for nidx_d in range(0, dim_data_in):
            y = data_in[:, nidx_d]
            y_out = np.interp(t_out, t_in, y)
            data_out[:, nidx_d] = y_out
    else:
        data_out = []

    return data_out


def interpolate_traj(traj_in, alpha=2):
    """ Interpolates trajectory.
    :param traj_in: input trajectory [x, y, theta].
    :param alpha: interpolation multiplier
    """
    if len(traj_in) > 0:
        traj_in = make_numpy_array(traj_in, keep_1dim=False)
        len_data_in = traj_in.shape[0]
        dim_data_in = traj_in.shape[1]
        t_in = np.arange(0, len_data_in)

        len_data_out = 1 + (len_data_in - 1) * alpha
        t_out = np.linspace(start=t_in[0], stop=t_in[-1], num=len_data_out)

        traj_out = np.zeros((len_data_out, dim_data_in), dtype=np.float32)
        for nidx_d in range(0, dim_data_in - 1):
            y = traj_in[:, nidx_d]
            y_out = np.interp(t_out, t_in, y)
            traj_out[:, nidx_d] = y_out

        for nidx_i0 in range(len_data_in - 1):
            theta0 = traj_in[nidx_i0, -1]
            theta1 = traj_in[nidx_i0 + 1, -1]
            diff_theta = get_diff_angles(theta1, theta0)

            for nidx_i1 in range(alpha):
                theta_new = theta0 + (nidx_i1 / alpha) * diff_theta
                traj_out[alpha * nidx_i0 + nidx_i1, -1] = theta_new

        traj_out[-1, -1] = traj_in[-1, -1]
    else:
        traj_out = []

    return traj_out


def apply_gaussian_kde_2d_naive(x, xref, sigma):
    """ Applies (naive) Gaussian kde.
    """
    x = make_numpy_array(x, keep_1dim=False)
    xref = make_numpy_array(xref, keep_1dim=False)

    len_x, len_xref = x.shape[0], xref.shape[0]

    z = np.zeros((len_x,), dtype=np.float32)

    for nidx_x in range(0, len_x):
        x_sel = x[nidx_x, :]
        x_sel = np.reshape(x_sel, (1, -1))
        x_sel = np.tile(x_sel, (len_xref, 1))

        diff_tmp = x_sel - xref
        dist_sq_tmp = np.sum(diff_tmp * diff_tmp, axis=1)
        exponent_tmp = -1.0 * dist_sq_tmp / (2.0 * sigma * sigma)
        density_tmp = np.exp(exponent_tmp)
        density_tmp = np.reshape(density_tmp, -1)

        z[nidx_x] = np.sum(density_tmp)

    return z
