# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import math
import numpy as np


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
