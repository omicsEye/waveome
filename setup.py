import setuptools

setuptools.setup(
    name="waveome",
    version="0.1.0",
    author="Allen Ross",
    author_email="allenross@gwu.edu",
    description="Automated longitudinal data analysis using Gaussian processes.",
    url="https://github.com/omicsEye/waveome",
    project_urls={
        "Bug Tracker": "https://github.com/omicsEye/waveome/issues"
    },
    license="MIT",
    packages=["waveome"],
    install_requires=[
        # More flexible TensorFlow requirements (but pinned for TF prob and GPflow compatibility)
        "tensorflow>=2.12.0,<2.16.0",
        "tensorflow_probability>=0.20.0,<0.24.0",
        "gpflow==2.9.1",  # Pin to specific version to avoid pkg_resources warning
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ray[default]",
        "scipy>=1.11.0",
        "joblib",
        "seaborn",
        "matplotlib",
        "tqdm",
        "setuptools>=65.5.1,<66.0.0",  # Add this to prevent pkg_resources deprecation warning
        "psutil"
    ],
    python_requires=">=3.9,<3.12",
)
