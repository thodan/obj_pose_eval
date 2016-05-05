#!/usr/bin/env python

# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A synthetic example of a rotating cup used for evaluation in the research report.

import numpy as np
import matplotlib.pyplot as plt
import transformations as trans
import inout
from pose_error import ad, cou, cpd, tre, vsd
from pose_error.utils import renderer

# Load object model
model_path = 'cup.ply'
model = inout.read_ply(model_path)

# Camera parameters
K = np.eye(3)
K[0, 0] = 500.0 # fx
K[1, 1] = 500.0 # fy
K[0, 2] = 250.0 # cx
K[1, 2] = 250.0 # cy
im_size = (500, 500)

# Calculate the poses of the rotating cup
poses = []
alpha_range = np.linspace(0, 360, 361)
for alpha in alpha_range:
    def d2r(d): return np.pi * float(d) / 180.0 # Degrees to radians
    R = trans.rotation_matrix(d2r(alpha), [0, 1, 0])[:3, :3] # Rotation around Y
    R = trans.rotation_matrix(d2r(30), [1, 0, 0])[:3, :3].dot(R) # Rotation around X
    t = np.array([0.0, 0.0, 180]).reshape((3, 1))

    # Flip Y axis (model coordinate system -> OpenCV coordinate system)
    R = trans.rotation_matrix(np.pi, [1, 0, 0])[:3, :3].dot(R)
    poses.append({'R': R, 't': t})

# Set and render the ground truth pose
gt_id = 90 # ID of the ground truth pose
pose_gt = poses[gt_id]
pose_gt_indis_set_ids = range(55, 126) # IDs of poses indistinguishable from the GT pose
pose_gt_indis_set = [poses[i] for i in pose_gt_indis_set_ids]
depth_gt = renderer.render_model(model, im_size, K, pose_gt['R'], pose_gt['t'], 100, 2000, mode='depth')

# Synthesize the test depth image
depth_test = np.array(depth_gt)
depth_test[depth_test == 0] = 1000

# Available errors: 'vsd', 'acpd', 'mcpd', 'add', 'adi', 'te', 're', 'cou'
# Errors to be calculated:
errs_active = ['vsd']

# Calculate the pose errors
errs = {err: [] for err in errs_active}
for pose_id, pose in enumerate(poses):
    print 'Processing pose:', pose_id

    # Determine the set of poses that are indistinguishable from the current pose
    if 55 <= pose_id <= 125:
        pose_indis_set = pose_gt_indis_set
    else:
        pose_indis_set = [pose]

    # Visible Surface Discrepancy
    if 'vsd' in errs_active:
        delta = 3
        tau = 30
        errs['vsd'].append(vsd.vsd(pose, pose_gt, model, depth_test, delta, tau, K))

    # Average Corresponding Point Distance
    if 'acpd' in errs_active:
        errs['acpd'].append(cpd.acpd(pose_indis_set, pose_gt_indis_set, model))

    # Maximum Corresponding Point Distance
    if 'mcpd' in errs_active:
        errs['mcpd'].append(cpd.mcpd(pose_indis_set, pose_gt_indis_set, model))

    # Average Distance of Model Points for objects with no indistinguishable views
    if 'add' in errs_active:
        errs['add'].append(ad.add(pose, pose_gt, model))

    # Average Distance of Model Points for objects with indistinguishable views
    if 'adi' in errs_active:
        errs['adi'].append(ad.adi(pose, pose_gt, model))

    # Translational Error
    if 'te' in errs_active:
        errs['te'].append(tre.te(pose['t'], pose_gt['t']))

    # Rotational Error
    if 're' in errs_active:
        errs['re'].append(tre.re(pose['R'], pose_gt['R']))

    # Complement over Union
    if 'cou' in errs_active:
        errs['cou'].append(cou.cou(pose, pose_gt, model, im_size, K))

# Plot the errors
for err_name in errs_active:
    plt.figure()
    plt.plot(errs[err_name], c='r', lw='3')
    plt.xlabel('Pose ID')
    plt.ylabel(err_name)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
plt.show()
