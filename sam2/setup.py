from setuptools import setup, find_packages

setup(
    name="sam2",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "numpy>=1.24.4",
        "tqdm>=4.66.1",
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
        "pillow>=9.4.0",
    ],
    python_requires=">=3.10",
)
