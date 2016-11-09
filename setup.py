from distutils.core import setup

setup(
    name='obj_pose_eval',
    version='1.0',
    py_modules=['obj_pose_eval'],
    license='The MIT License',
    long_description=open('README.md').read(),
    install_requires=['scipy', 'numpy', 'glumpy', 'matplotlib']
)
