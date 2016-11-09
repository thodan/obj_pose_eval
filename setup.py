from distutils.core import setup

setup(
    name='obj_pose_eval',
    version='1.0',
    description='Tools for evaluation of 6D object pose estimation',
    long_description=open('README.md').read(),
    author='Tomas Hodan',
    author_email='hodantom@cmp.felk.cvut.cz',
    url='https://github.com/thodan/obj_pose_eval',
    packages=['obj_pose_eval'],
    license='The MIT License',
    install_requires=['scipy', 'numpy', 'glumpy', 'matplotlib']
)
