# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
from pose_error.utils import misc


def acpd(poses_est, poses_gt, model):
    """
    Average Corresponding Point Distance.

    :param poses_est: List of poses that are indistinguishable from the
    estimated pose. Each pose is given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param poses_gt: List of poses that are indistinguishable from the
    ground truth pose (format as for poses_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    e = -1
    for pose_est in poses_est:
        pts_est = misc.transform_pts_Rt(model['pts'], pose_est['R'], pose_est['t'])
        for pose_gt in poses_gt:
            pts_gt = misc.transform_pts_Rt(model['pts'], pose_gt['R'], pose_gt['t'])
            dist = np.linalg.norm(pts_gt - pts_est, axis=1).mean()
            if dist < e or e == -1:
                e = dist
    return e


def mcpd(poses_est, poses_gt, model):
    """
    Maximum Corresponding Point Distance.

    :param poses_est: List of poses that are indistinguishable from the
    estimated pose. Each pose is given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param poses_gt: List of poses that are indistinguishable from the
    ground truth pose (format as for poses_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    e = -1
    for pose_est in poses_est:
        pts_est = misc.transform_pts_Rt(model['pts'], pose_est['R'], pose_est['t'])
        for pose_gt in poses_gt:
            pts_gt = misc.transform_pts_Rt(model['pts'], pose_gt['R'], pose_gt['t'])
            dist = np.linalg.norm(pts_gt - pts_est, axis=1).max()
            if dist < e or e == -1:
                e = dist
    return e
