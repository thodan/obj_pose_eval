### Tools for Evaluation of 6D Object Pose Estimation

The tools are implemented in Python 2.7 and require libraries SciPy (0.18.1), NumPy (1.11.2) and Glumpy (1.0.6). Samples additionally require Matplotlib (1.5.3) and OpenCV (3.1). Tested library versions are in the brackets.

The evaluation methodology is described in:
```
T. Hodaň, J. Matas, Š. Obdržálek,
On Evaluation of 6D Object Pose Estimation,
European Conference on Computer Vision Workshops (ECCVW) 2016, Amsterdam
http://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
```

**Installation:**
```
pip install git+https://github.com/thodan/obj_pose_eval.git
```

This command will install also the required libraries, with exception of OpenCV. For instructions on how to install OpenCV with the Python interface, see: https://breakthrough.github.io/Installing-OpenCV

**Samples:**

- samples/**eval_rotating_cup.py**: Evaluates the pose error functions.
- samples/**render_demo.py**: Renders color/depth image of a 3D mesh model.
- samples/**show_visibility_dresden.py**: Estimates visibility masks in images from the ICCV2015 Occluded Object Dataset - http://cvlab-dresden.de/iccv2015-occlusion-challenge.

Author: **Tomas Hodan** (hodantom@cmp.felk.cvut.cz, http://www.hodan.xyz), Center for Machine Perception, Czech Technical University in Prague
