from setuptools import setup, find_packages

setup(
    name="metal_covariance_matrix",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "biopython",
        "tqdm",
        "dendropy",
        "pyvolve",
        "matplotlib",
        "scikit-learn"
    ],
)