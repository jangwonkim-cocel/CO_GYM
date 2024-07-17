from setuptools import find_packages, setup
import pathlib
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='co_gym',
    author='Jangwon Kim, Jaehyung Cho',
    author_email='jangwonkim@postech.ac.kr, jaehyungcho@postech.ac.kr',
    #url="https://github.com/",
    License='MIT',
    version='0.2.1',  # Under Developing
    packages=find_packages(include=['.', 'co_gym']),
    include_package_data=True,
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
            'python>=3.8, <3.11'
            'gymnasium>=0.17.2',
            'torch>=1.7.0',
            'numpy>=1.16.0',
            'wandb',
            'pyyaml',
            'pandas',
            'mujoco-python-viewer'
      ],
)
