from setuptools import setup, find_packages

setup(
    name="selective-decision-feedback",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    author="Ahmed",
    description="Pilot-aided + Decision-Directed Channel Estimation with Diffusion",
)