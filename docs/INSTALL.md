# Installation Guide for waveome

This guide provides detailed instructions for installing `waveome` on different platforms.  
The package depends on [TensorFlow](https://www.tensorflow.org/), [GPflow](https://www.gpflow.org/), and several scientific Python libraries, so we recommend using conda to manage environments.


### 1. Prerequisites

- Python: version 3.9–3.11 (TensorFlow does not yet support 3.12+)
- conda: [Install Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/)
- git: required to clone/install directly from GitHub

Check installations:

```bash
conda --version
git --version
```

### 2. Create and activate a conda environment

We recommend a clean environment for reproducibility:
```bash
conda create -n waveome_env python=3.11
conda activate waveome_env
```

### 3. Install waveome
Install from PyPi:
```bash
pip install waveome
```
or install directly from GitHub:
```bash
pip install git+https://github.com/omicsEye/waveome
```

### 4. Optional: Jupyter Notebook setup
```bash
conda install jupyter ipykernel
python -m ipykernel install --user --name=waveome_env
```
If you plan to run tutorials such as `waveome_overview.ipynb`.

### 5. Platform-specific notes
#### Windows
* Ensure [Microsoft Visual C++ 14.0+](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is installed.
* Use [Git for Windows](https://gitforwindows.org/) to access the `git` command.
#### macOS (M1/M2)
* TensorFlow on Apple Silicon sometimes requires an extra dependency. If you face issues importing `grpcio`, install it first:
```bash
conda install -c conda-forge grpcio
```
Then proceed with Jupyter installation if needed.

### 6. Test your installation
Run the following to check that `waveome` is available:
```bash
python -c "import waveome"
```
This should result in no errors.

### 7. Troubleshooting
* If conda is not recognized, ensure it is added to your PATH (reinstall if needed).
* For TensorFlow GPU support see the [TensorFlow install guide](https://www.tensorflow.org/install) (`waveome` works on CPU by default, and rigorous testing on GPU has not yet been performed).
* Report issues here: https://github.com/omicsEye/waveome/issues
