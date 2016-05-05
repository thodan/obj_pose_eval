# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
from scipy import spatial
from pose_error.utils import misc


def add(pose_est, pose_gt, model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param pose_est: Estimated pose given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param pose_gt: The ground truth pose given by a dictionary (as pose_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_gt = misc.transform_pts_Rt(model['pts'], pose_gt['R'], pose_gt['t'])
    pts_est = misc.transform_pts_Rt(model['pts'], pose_est['R'], pose_est['t'])
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(pose_est, pose_gt, model):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param pose_est: Estimated pose given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param pose_gt: The ground truth pose given by a dictionary (as pose_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_gt = misc.transform_pts_Rt(model['pts'], pose_gt['R'], pose_gt['t'])
    pts_est = misc.transform_pts_Rt(model['pts'], pose_est['R'], pose_est['t'])

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e
