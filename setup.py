import setuptools

setuptools.setup(
    name='waveome',
    version='0.0.2',
    author='Allen Ross',
    author_email='allenross@gwu.edu',
    description='Automated longitduinal data analysis using Gaussian processes',
    url='https://github.com/omicsEye/waveome',
    project_urls = {
        "Bug Tracker": "https://github.com/omicsEye/waveome/issues"
    },
    license='MIT',
    packages=['waveome'],
    install_requires=[
        'gpflow==2.9.0', 
        'tensorflow_probability==0.21.0', 
        'tensorflow==2.13.0',
        # "tensorflow-macos; platform_machine=='arm64'",
        'pandas', 
        'numpy', 
        'scipy',
        'joblib',
        'seaborn',
        'matplotlib',
        'tqdm'
    ],
)