from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name='ConfidNet',
    author='Charles Corbiere',
    url='https://github.com/valeoai/ConfidNet',
    ext_modules=[CppExtension('confignet.structured_map_ranking_loss',
                              ['cpp_loss/structured_map_ranking_loss.cpp'])],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['pyyaml', 'coloredlogs',
                      'torchsummary', 'verboselogs',
                      'future',
                      'setuptools', 'tqdm'],
    packages=find_packages())
