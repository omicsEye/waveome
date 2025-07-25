import setuptools

setuptools.setup(
    name="waveome",
    version="0.0.3",
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
        
        # Need to pin TF and TF prob together
        # Pin for python 3.11.8 - not mac specific
        "tensorflow==2.15.1",
        "tensorflow_probability==0.23.0",

        # Main package - doesn't handle TF well at install
        "gpflow==2.9.1",
        # "tf-keras", # Don't use this for the time being with keras opts

        # # Pin for python 3.8.12 (tensorflow-metal GPU)
        # "tensorflow==2.13.0",
        # "tensorflow-metal==1.0.1",
        # "tensorflow_probability==0.21.0",

        "pandas==2.1.4",  # Need to pin this for numpy errors
        "numpy==1.26.1",  # Need to pin this for TF import errors
        "ray[default]",
        "scipy==1.11.2",  # Need to pin this for TF import errors
        "joblib",  # Switch out for ray
        "seaborn",
        "matplotlib",
        "tqdm"
    ],
)
