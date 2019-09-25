from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='ap_ranking_loss',
      ext_modules=[CppExtension('structured_map_ranking_loss', ['structured_map_ranking_loss.cpp'])],
      cmdclass={'build_ext': BuildExtension})
