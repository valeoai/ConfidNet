from setuptools import find_packages
from setuptools import setup

setup(name='ConfidNet',
      author='Charles Corbiere',
      url='https://github.com/valeoai/ConfidNet',
      install_requires=['pyyaml', 'coloredlogs',
                        'torchsummary', 'verboselogs',
                        'incense', 'future',
                        'setuptools', 'tqdm'],
      packages=find_packages())