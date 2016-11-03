#!/usr/bin/env python

# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague


def match_poses(ests, gts, ests_errs, error_thresh, prune_ests=True):

    # Sort the estimated poses by decreasing confidence score
    ests_sort_inds = [i[0] for i in sorted(enumerate(ests),
                                           key=lambda k: k[1]['score'],
                                           reverse=True)]

    # If there are more estimated poses than the specified number of instances,
    # keep only the poses with the highest confidence score
    if prune_ests:
        ests_sort_inds = ests_sort_inds[:len(gts)]

    # Mask of availability of ground truth poses
    poses_gt_avail = [True for _ in range(len(gts))]

    # Greedily match the estimated poses with the most similar ground truth poses
    matches = []
    for est_id in ests_sort_inds:
        best_gt_id = -1
        best_gt_error = -1
        for gt_id in range(len(gts)):
            if poses_gt_avail[gt_id]:
                if ests_errs[est_id][gt_id] < best_gt_error or best_gt_id == -1:
                    best_gt_id = gt_id
                    best_gt_error = ests_errs[est_id][gt_id]

        if best_gt_id != -1 and best_gt_error < error_thresh:
            best_gt_error_norm = best_gt_error / float(error_thresh)
            matches.append({'est_id': est_id, 'gt_id': best_gt_id,
                            'error': best_gt_error, 'error_norm': best_gt_error_norm})
            poses_gt_avail[best_gt_id] = False

    return matches
