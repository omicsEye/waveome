import setuptools

setuptools.setup(
    name='waveome',
    version='0.0.1',
    author='Allen Ross',
    author_email='allenross@gwu.edu',
    description='Automated longitduinal data analysis using Gaussian processes',
    url='https://github.com/omicsEye/waveome',
    project_urls = {
        "Bug Tracker": "https://github.com/omicsEye/waveome/issues"
    },
    license='MIT',
    packages=['waveome'],
    install_requires=['gpflow', 'pandas', 'numpy', 'scipy'],
)