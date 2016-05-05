# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
from utils import renderer


def cou(pose_est, pose_gt, model, im_size, K):
    """
    Complement over Union, i.e. the inverse of the Intersection over Union used
    in the PASCAL VOC challenge - by Everingham et al. (IJCV 2010).

    :param pose_est: Estimated pose given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param pose_gt: The ground truth pose given by a dictionary (as pose_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param im_size: Test image size.
    :param K: Camera matrix.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    # Render depth images of the model in the estimated and the ground truth pose
    d_est = renderer.render_model(model, im_size, K, pose_est['R'], pose_est['t'],
                                  clip_near=100, clip_far=10000, mode='depth')

    d_gt = renderer.render_model(model, im_size, K, pose_gt['R'], pose_gt['t'],
                                 clip_near=100, clip_far=10000, mode='depth')

    # Masks of the rendered model and their intersection and union
    mask_est = d_est > 0
    mask_gt = d_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    e = 1.0 - inter.sum() / float(union.sum())
    return e
