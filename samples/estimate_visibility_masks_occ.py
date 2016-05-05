#!/usr/bin/env python

# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A script for estimation of visibility masks in images from the ICCV2015 Occluded
# Object Challenge [1]. Besides the dataset [2], please download also the object
# models in PLY format from [3] and put them into subfolder "model_ply" of
# the main dataset folder "OcclusionChallengeICCV2015". You will also need to
# set data_basepath below.
#
# [1] http://cvlab-dresden.de/iccv2015-occlusion-challenge
# [2] https://cloudstore.zih.tu-dresden.de/public.php?service=files&t=a65ec05fedd4890ae8ced82dfcf92ad8
# [3] http://cmp.felk.cvut.cz/~hodanto2/store/OcclusionChallengeICCV2015_models_ply.zip

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import inout
from pose_error import vsd
from pose_error.utils import misc, renderer

# Path to the OcclusionChallengeICCV2015 folder:
#-------------------------------------------------------------------------------
data_basepath = 'YOUR_PATH_HERE/OcclusionChallengeICCV2015'
#-------------------------------------------------------------------------------

objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
calib_fpath = os.path.join(data_basepath, 'calib.yml')
model_fpath_mask = os.path.join(data_basepath, 'models_ply', '{0}.ply')
rgb_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'rgb_noseg', 'color_{0}.png')
depth_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'depth_noseg', 'depth_{0}.png')
gt_poses_mask = os.path.join(data_basepath, 'poses', '{0}', '*.txt')

# Camera parameters
im_size = (640, 480)
K = np.array([[572.41140, 0, 325.26110],
              [0, 573.57043, 242.04899],
              [0, 0, 0]])

# Load models and ground truth poses
models = []
gt_poses = []
for obj in objs:
    print 'Loading data:', obj
    model_fpath = model_fpath_mask.format(obj)
    models.append(inout.read_ply(model_fpath))

    gt_fpaths = sorted(glob.glob(gt_poses_mask.format(obj)))
    gt_poses_obj = []
    for gt_fpath in gt_fpaths:
        gt_poses_obj.append(inout.read_gt_pose_OcclusionChallengeICCV2015(gt_fpath))
    gt_poses.append(gt_poses_obj)

# Prepare figure for visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
cb_dist_diff = None
cb_dist = None

# Loop over images
im_ids = range(len(gt_poses[0])) #range(100)
for im_id in im_ids:
    im_id_str = str(im_id).zfill(5)
    print 'Processing image:', im_id

    # Load the RGB and the depth image
    rgb_fpath = rgb_fpath_mask.format(im_id_str)
    rgb = cv2.imread(rgb_fpath, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth_fpath = depth_fpath_mask.format(im_id_str)
    depth = cv2.imread(depth_fpath, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Convert the input depth image to a distance image
    dist = misc.depth_im_to_dist_im(depth, K)

    for obj_id, obj_name in enumerate(objs):
        pose = gt_poses[obj_id][int(im_id)]
        if pose['R'].size != 0 and pose['t'].size != 0:

            # Render the object model
            depth_ren_gt = renderer.render_model(
                models[obj_id], im_size, K, pose['R'], pose['t'], 0.1, 2.0,
                surf_color=(0.0, 1.0, 0.0), mode='depth')
            depth_ren_gt *= 1000 # Convert the rendered depth map to [mm]

            # Convert the input depth image to a distance image
            dist_ren_gt = misc.depth_im_to_dist_im(depth_ren_gt, K)

            # Estimate the visibility mask
            delta = 15 # [mm]
            visib_mask = vsd.estimate_visibility_mask(dist, dist_ren_gt, delta)

            # Get the non-visibility mask
            nonvisib_mask = np.logical_and(~visib_mask, dist_ren_gt > 0)

            # Difference between the test and the rendered distance image
            dist_diff = dist_ren_gt.astype(np.float32) - dist.astype(np.float32)
            dist_diff *= np.logical_and(dist > 0, dist_ren_gt > 0).astype(np.float32)

            # Visualization
            #-------------------------------------------------------------------
            # Clear axes
            for ax in axes.flatten():
                ax.clear()

            # Get colored visib/nonvisib mask
            ch_zeros = np.zeros(nonvisib_mask.shape)
            visib_mask_rgb = np.dstack((ch_zeros, visib_mask, ch_zeros))
            nonvisib_mask_rgb = np.dstack((nonvisib_mask, ch_zeros, ch_zeros))
            mask_rgb = ((visib_mask_rgb + nonvisib_mask_rgb) * 255).astype(np.float)

            # Input RGB image with overlaid masks
            vis_rgb = rgb.astype(np.float) * 0.5 + mask_rgb * 0.5
            axes[0, 0].imshow(vis_rgb.astype(np.uint8))
            axes[0, 0].set_title('Mask on RGB image (green = visible, red = invisible)')

            # Input distance image with overlaid masks
            vis_dist_bg = np.maximum(255 * dist / 1500.0, np.ones(dist.shape))
            vis_dist_bg = np.dstack((vis_dist_bg, vis_dist_bg, vis_dist_bg))
            vis_dist = vis_dist_bg * 0.5 + mask_rgb * 0.5
            axes[1, 0].imshow(vis_dist.astype(np.uint8))
            axes[1, 0].set_title('Mask on distance image (green = visible, red = invisible)')

            # Input distance image
            ai_dist = axes[0, 1].matshow(dist, cmap='inferno')
            axes[0, 1].set_title('Input distance image [mm]')
            if not cb_dist:
                cb_dist = fig.colorbar(ai_dist, ax=axes[0, 1])
            else:
                cb_dist.update_bruteforce(ai_dist)

            # Distance difference
            ai_dist_diff = axes[1, 1].matshow(dist_diff, cmap='inferno')
            axes[1, 1].set_title('Difference of distances [mm]')
            if not cb_dist_diff:
                cb_dist_diff = fig.colorbar(ai_dist_diff, ax=axes[1, 1])
            else:
                cb_dist_diff.update_bruteforce(ai_dist_diff)

            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.15, wspace=0.15)
            plt.draw()
            plt.waitforbuttonpress()
