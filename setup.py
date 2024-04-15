import setuptools

setuptools.setup(
    name="waveome",
    version="0.0.2",
    author="Allen Ross",
    author_email="allenross@gwu.edu",
    description="""
        Automated longitudinal data analysis using Gaussian processes.
    """,
    url="https://github.com/omicsEye/waveome",
    project_urls={
        "Bug Tracker": "https://github.com/omicsEye/waveome/issues"
    },
    license="MIT",
    packages=["waveome"],
    install_requires=[
        # Main package - doesn't handle TF well at install
        "gpflow==2.9.1",
        # Need to pin TF and TF prob together
        "tensorflow_probability==0.23.0",
        "tensorflow==2.15.1",
        "pandas",
        "numpy",
        "scipy",
        "joblib",
        "seaborn",
        "matplotlib",
        "tqdm"
    ],
)
