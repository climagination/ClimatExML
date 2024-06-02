from setuptools import setup

setup(
    name="ClimatExML",
    version="0.1.0",
    author="Nic Annau",
    author_email="nicannau@gmail.com ",
    packages=["ClimatExML"],
    license="LICENSE",
    description="Python module to to pre-process ClimatEx deep learning data.",
    long_description=open("README.md").read(),
    install_requires=[
        "pytorch-lightning",
        "lightning",
        "torch",
        "hydra-core",
        "pytorch_msssim",
        "torchvision",
        "matplotlib",
        "pydantic",
        "comet-ml"
    ],
)
