# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np
from utils import misc, renderer


def estimate_visibility_mask(d_test, d_model, delta):
    """
    Estimation of visibility mask.

    :param d_test: Distance image of the test scene.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :return: Visibility mask.
    """
    assert(d_test.shape == d_model.shape)
    mask_valid = np.logical_and(d_test > 0, d_model > 0)

    d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    return visib_mask


def vsd(pose_est, pose_gt, model, depth_test, delta, tau, K):
    """
    Visible Surface Discrepancy.

    :param pose_est: Estimated pose given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param pose_gt: The ground truth pose given by a dictionary (as pose_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    im_size = (depth_test.shape[1], depth_test.shape[0])

    # Render depth images of the model in the estimated and the ground truth pose
    depth_est = renderer.render_model(model, im_size, K, pose_est['R'], pose_est['t'],
                                      clip_near=100, clip_far=10000, mode='depth')

    depth_gt = renderer.render_model(model, im_size, K, pose_gt['R'], pose_gt['t'],
                                     clip_near=100, clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = estimate_visibility_mask(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = estimate_visibility_mask(dist_test, dist_est, delta)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, dist_est > 0))

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    costs *= (1.0 / tau)
    costs[costs > 1.0] = 1.0

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    e = (costs.sum() + visib_comp_count) / visib_union_count

    return e
