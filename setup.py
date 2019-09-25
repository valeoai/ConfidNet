from setuptools import find_packages
from setuptools import setup

setup(name='ConfidNet',
      install_requires=['pyyaml', 'coloredlogs',
                        'torchsummary', 'verboselogs',
                        'incense', 'future',
                        'setuptools', 'tqdm'],
      packages=find_packages())