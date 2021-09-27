"""Install script for setuptools."""

from setuptools import setup


setup(
    name='cpmolgan',
    version='0.1',
    packages=['cpmolgan'],
    url='https://github.com/pamzgt/CPMolGAN',
    license='MIT License',
    author='Paula A. Marin Zapata, Oscar Méndez-Lucio',
    author_email='paula.marinzapata@bayer.com, oscar.mendez.lucio@gmail.com',
    description='Cell morphology-guided de novo hit design by conditioning GANs on phenotypic image features'
)