#!/usr/bin/env python

# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Rendering demo.

import matplotlib.pyplot as plt
import numpy as np

from obj_pose_eval import renderer, inout, transform

# Load object model
model_path = 'cup.ply'
model = inout.load_ply(model_path)

# Camera parameters
K = np.eye(3)
K[0, 0] = 500.0 # fx
K[1, 1] = 500.0 # fy
K[0, 2] = 250.0 # cx
K[1, 2] = 250.0 # cy
im_size = (500, 500)

R = transform.rotation_matrix(np.pi, (1, 0, 0))[:3, :3]
t = np.array([[0, 0, 150]]).T

rgb, depth = renderer.render(model, im_size, K, R, t, 100, 2000, mode='rgb+depth')
# depth = renderer.render(model, im_size, K, R, t, 100, 2000, mode='depth')
# rgb = renderer.render(model, im_size, K, R, t, 100, 2000, mode='rgb')

plt.imshow(rgb)
plt.title('Rendered color image')

plt.matshow(depth)
plt.colorbar()
plt.title('Rendered depth image')

plt.show()
