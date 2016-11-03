# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import math
import numpy as np
from scipy import spatial
import renderer
import misc
import visibility

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
    depth_est = renderer.render(model, im_size, K, pose_est['R'], pose_est['t'],
                                clip_near=100, clip_far=10000, mode='depth')

    depth_gt = renderer.render(model, im_size, K, pose_gt['R'], pose_gt['t'],
                               clip_near=100, clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    costs *= (1.0 / tau)
    costs[costs > 1.0] = 1.0

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / visib_union_count
    else:
        e = 1.0
    return e

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
    d_est = renderer.render(model, im_size, K, pose_est['R'], pose_est['t'],
                            clip_near=100, clip_far=10000, mode='depth')

    d_gt = renderer.render(model, im_size, K, pose_gt['R'], pose_gt['t'],
                           clip_near=100, clip_far=10000, mode='depth')

    # Masks of the rendered model and their intersection and union
    mask_est = d_est > 0
    mask_gt = d_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e

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

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose given by a 3x1 vector.
    :param t_gt: Translation element of the ground truth pose given by a 3x1 vector.
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose given by a 3x3 matrix.
    :param R_gt: Rotational element of the ground truth pose given by a 3x3 matrix.
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    return error
