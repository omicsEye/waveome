# waveome #
## Overview
<img style="float: right;" width="200" height="200" src="./figures/hex-waveome.png">

This repository houses code for the _Waveome_ package - an easy to use and powerful Python library that analyzes longitudinal data using Gaussian processes.

<!-- ![waveome logo](./figures/hex-waveome.png) -->
<!-- <img src="./figures/hex-waveome.png" width="200" height="200"> -->


***waveome*** is a computational method for longitudinal data analysis particularly  
to characterize and identify temporal dynamics of omics and clinical variables in association with
the phenotype of interest.

---
**Key features:**

* **Generality:** *waveome* is a new computational tool for identifying temporal dynamics
  significantly associated with phenotypes of interest.
* **Validation:** A comprehensive evaluation of waveome performance using synthetic
  data generation with known ground truth for genotype-phenotype association testing.
* **Interpretation:** Rather than checking all possible kernels (dynamics), _waveome_ prioritizes only comprehensive and 
interpretable  prior kernels.
* **Elegance:** User-friendly, open-source software allowing for high-quality visualization
  and statistical tests.
* **Optimization:** Since omics data are often very high dimensional, all modules have been written and benchmarked for computing time.
* **Documentation:** Open-source GitHub repository of code complete with tutorials and a wide range of
  real-world applications.

---
**Citation:**

Allen Ross, Ali Reza Taheriouyn, Jason Llyod-Price, Ali Rahnavard (2024).
**_waveome_: characterizing temporal dynamics of metabolites in longitudinal studies**, https://github.com/omicsEye/waveome/.

---

# waveome user manual #

## Contents ##

* [Features](#features)
* [waveome](#waveome)
    * [Installation](#installation)
        * [Windows Linux Mac](#windows-linux-mac)
        * [Apple M1/M2 MAC](#apple-m1m2-mac)
* [Getting Started with waveome](#getting-started-with-waveome)
    * [Test waveome](#test-waveome)
    * [Options](#options)
    * [Input](#input)
    * [Output](#output)
    * [Demo](#demo)
    * [Tutorial](#tutorial)
* [Applications](#applications)
    * [*waveome* identifies important dynamics in metabolites and associations with metadata](#ihmp)
    * [Application 2](#application2)
    * [Application 3](#application3)
    * [Application 4](#application4)
    * [Application 5](#application5)
* [Support](#support)
------------------------------------------------------------------------------------------------------------------------------
# Features #
1. Generic software that can handle any kind of sequencing data and phenotypes
2. One place to do all analysis and producing high-quality visualizations
3. Optimized computation
4. User-friendly software
5. Provides temporal dynamics and associated omics features and metadata



## Installation ##
## General Usage
Please see the `waveome_overview.ipynb` for an example of the package modeling simulated data.

## Installation
* First install *conda*  
  Go to the [Anaconda website](https://www.anaconda.com/) and download the latest version for your operating system.
* For Windows users: do not forget to add `conda` to your system `path`
* Second is to check for conda availability  
  open a terminal (or command line for Windows users) and run:
```
conda --version
```
it should out put something like:
```
conda 4.9.2
```
if not, you must make *conda* available to your system for further steps.
if you have problems adding conda to PATH, you can find instructions
[here](https://docs.anaconda.com/anaconda/user-guide/faq/).
### Windows Linux Mac ###
If you are using an **Apple M1/M2 MAC** please go to the [Apple M1/M2 MAC](#apple-m1m2-mac) for installation
instructions.  
If you have a working conda on your system, you can safely skip to step three.  
If you are using windows, please make sure you have both git and Microsoft Visual C++ 14.0 or greater installed.
install [git](https://gitforwindows.org/)
[Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
In case you face issues with this step, [this link](https://github.com/pycaret/pycaret/issues/1254) may help you.
1) Create a new conda environment (let's call it waveome_env) with the following command:
```
conda create --name waveome_env python=3.9
```
2) Activate your conda environment:
```commandline
conda activate waveome_env 
```
3) Install *waveome*:
   you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/waveome
```
### Apple M1/M2 MAC ###
1) Update/install Xcode Command Line Tools
  ```commandline
  xcode-select --install
  ```

2) Close the current terminal and open a new terminal
3) Create a new conda environment (let's call it waveome_env) with the following command:
  ```commandline
  conda create --name waveome_env python=3.8.12
  ```
4) Activate the conda environment
  ```commandline
  conda activate waveome_env
  ```
5) Warning: If you are using an M1 Mac then before "pip install" you should run:
   `conda install -c conda-forge grpcio`
6) Finally, install *waveome*:

you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/waveome
or
and then pip install _waveome_ using    
pip install git+https://github.com/omicsEye/waveome.git 
```



## Jupyter Notebook
If you would like to run `waveome_overview.ipynb` then you should also set up a Jupyter kernel for the new waveome environment.
This can be done with
`conda install ipykernel`
and then
`python -m ipykernel install --user --name=waveome`.


-----------------------------------------------------------------------------------------------------------------------

# Getting Started with waveome #

## Test waveome ##

To test if waveome is installed correctly, you may run the following command in the terminal:

```#!cmd
waveome -h
```
Which yields waveome command line options.



## Options ##

```

```
## Input ##


## Output ##  


## Demo ##
```commandline
waveome -sf PATH_TO_SEQUENCE.FASTA -st aa -md PATH_TO_META_DATA.tsv -mv
 META_VARIABLE_NAME -a reg  -dth 0.15 --plot --write
```

## Tutorial ##
Multiple detailed jupyter notebook of _waveome_ implementation are available in the
[examples](https://github.com/omicsEye/waveome/tree/master/examples) and the
required data for the examples are also available in the
[data](https://github.com/omicsEye/waveome/tree/master/data) directory.


# Applications #
Here we try to use the **_waveome_** on different datasets and elaborate on the results.

<h2 id="opsin">
<i>waveome</i> identifies amino acids associated with color sensitivity
</h2>

![Opsins](https://github.com/omicsEye/waveome/blob/master/img/lite_mar/figure.png?raw=True)

Opsins are genes involved in light sensitivity and vision, and when coupled with a light-reactive chromophore, the
absorbance of the resulting photopigment dictates physiological phenotypes like color sensitivity. We analyzed the
amino acid sequence of rod opsins because previously published mutagenesis work established mechanistic connections
between 12 specific amino acid sites and phenotypes [Yokoyama et al. (2008)](https://doi.org/10.1073/pnas.0802426105).
Therefore, we hypothesized that machine learning approaches could predict known associations between amino acid sites
and absorbance phenotypes. We identified opsins expressed in
rod cells of vertebrates (mainly marine fishes) with absorption spectra measurements (λmax, the wavelength with the
highest absorption). The dataset contains 175 samples of opsin sequences. We next applied waveome on this
dataset to find the most important sites contributing to the variations of λmax.
This [Jupyter Notebook](https://github.com/omicsEye/waveome/blob/master/examples/continuous_phenotype_light_sensitivity.ipynb)
illustrates the steps.


<h2 id="hmp">
Novel insights of niche associations in the oral microbiome
</h2>

![hmp](https://github.com/omicsEye/waveome/blob/master/img/hmp/hmp.png?raw=True)  
Microbial species tend to adapt at the genome level to the niche in which they live. We hypothesize
that genes with essential functions change based on where microbial species live. Here we use microbial strain
representatives from stool metagenomics data of healthy adults from the
[Human Microbiome Project](https://doi.org/10.1038/nature11234). The input for waveome consists of 1) an MSA file
with 1006 rows, each a representative strain of a specific microbial species, here Haemophilus parainfluenzae, with
49839 lengths; and 2) labels for waveome prediction are body sites from which samples were collected.
This [Jupyter Notebook](https://github.com/omicsEye/waveome/blob/master/examples/discrete_phenotype_HMP.ipynb)
illustrates the steps.


<h2 id="covid">
<i>waveome</i> reveals important SARS-CoV-2 regions associated with Alpha and Delta variants
</h2>

![sarscov2](https://github.com/omicsEye/waveome/blob/master/img/sars_cov2/sarscov2.png?raw=True)
Variants occur with new mutations in the virus genome. Most mutations in the SARS-CoV-2 genome do not affect the
functioning of the virus. However, mutations in the spike protein of SARS-CoV-2, which binds to receptors on cells
lining the inside of the human nose, may make the virus easier to spread or affect how well vaccines protect people.
We are going to study the mutations in the spike protein of the sequences of Alpha (B.1.1.7): the first variant of
concern described in the United Kingdom (UK) in late December 2020 and Delta (B.1.617.2): first reported in India in
December 2020. We used the publicly available data from the [GSAID](https://gisaid.org/) and obtained 900 sequences
of spike protein region of Alpha (450 samples) and Delta (450 samples) variants. Then, we used waveome to analyze
the data and find the most important (predictive) positions in these sequences in terms of classifying the variants.
This
[Jupyter Notebook](https://github.com/omicsEye/waveome/blob/master/examples/discrete_phenotype_SARS_Cov2_variants.ipynb)
illustrates the steps.


<h2 id="hiv">
<i>waveome</i> identifies HIV regions with potentially important functions
</h2>

![sarscov2](https://github.com/omicsEye/waveome/blob/master/img/HIV/HIV3.png?raw=True)
Subtypes of the human immunodeficiency virus type 1 (HIV-1) group M are different in the envelope (Env) glycoproteins
of the virus. These parts of the virus are displayed on the surface of the virion and are targets for both neutralizing
antibody and cell-mediated immune responses. The third hypervariable domain (V3) of HIV-1 gp120 is a cysteine-bounded
loop structure usually composed of 105 nucleotides and labeled as the base (nu 1:26 and 75:105), stem
(nu 27:44 and 54:74), and turn (nu 45:53) regions [Lynch et al. (2009)](https://doi.org/10.1089%2Faid.2008.0219) .
Among all of the hyper-variable regions in gp120 (V1-V5), V3 is playing the main role in the virus infectivity
[Felsövályi et al. (2006)](https://doi.org/10.1089%2Faid.2006.22.703).
Here we useare using waveome to identify important regions in the V3 loop that are important in terms of associating
the V3 sequences V3 to subtypes B and C. We used the [Los Alamos HIV Database](www.hiv.lanl.gov) to gather the
nucleotide sequences of the V3 loop of subtypes B and C.
This [Jupyter Notebook](https://github.com/omicsEye/waveome/blob/master/examples/discrete_phenotype_HIV.ipynb)
illustrates the steps.

# Support #

* Please submit your questions or issues with the software at
  [Issues tracker](https://github.com/omicsEye/waveome/issues).
* For community discussions, questions, and issue reporting, please visit our forum [here](https://forum.omicseye.org/c/omics-downstream-analysis/waveome/12)





