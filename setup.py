# !/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages


setup(
    name='bigmouth',
    version='0.0.1',
    description='Render images from text',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/bigmouth',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'numpy',
        'pandas',
        'pytorch_pretrained_biggan',
        'torch',
        'torchvision',
    ]
)
