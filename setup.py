from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name="ConfidNet",
    version='0.1.0',
    author="Charles Corbiere",
    url="https://github.com/valeoai/ConfidNet",
    ext_modules=[
        CppExtension(
            "confidnet.structured_map_ranking_loss",
            ["confidnet/cpp_loss/structured_map_ranking_loss.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "pyyaml",
        "coloredlogs",
        "torchsummary",
        "verboselogs",
        "setuptools",
        "tqdm",
        "click",
        "scikit-learn",
    ],
    packages=find_packages(),
)
