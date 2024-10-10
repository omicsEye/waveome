# waveome #

## Overview

<img style="float: right;" width="200" height="200" src="./figures/hex-waveome.png">

This repository houses code for the _waveome_ package - an easy to use and powerful Python library that analyzes
longitudinal data using Gaussian processes.

<!-- ![waveome logo](./figures/hex-waveome.png) -->
<!-- <img src="./figures/hex-waveome.png" width="200" height="200"> -->

*waveome* is a computational method for longitudinal data analysis particularly  to characterize and identify temporal dynamics of omics and clinical variables in association with
the phenotype of interest. It employs the Gaussian processes as prior to implement a nonparametric
estimation for the dynamics of the underlying measurements.

---
**Key features:**

* **Generality:** *waveome* is a new computational tool for identifying temporal dynamics
  significantly associated with phenotypes of interest.
* **Validation:** A comprehensive evaluation of waveome performance using synthetic
  data generation with known ground truth for genotype-phenotype association testing.
* **Interpretation:** By prioritizing comprehensive and flexible kernel functions, _waveome_ significantly reduces
  computational costs.
* **Elegance:** User-friendly, open-source software allowing for high-quality visualization
  and statistical tests.
* **Optimization:** Since omics data are often very high dimensional, all modules have been written and benchmarked for
  computing time.
* **Documentation:** Open-source GitHub repository of code complete with tutorials and a wide range of
  real-world applications.

---
**Citation:**

Allen Ross, Ali Reza Taheriouyn, Jason Llyod-Price, Ali Rahnavard (2024).
**_waveome_: characterizing temporal dynamics of metabolites in longitudinal studies
**, https://github.com/omicsEye/waveome/.

---

# waveome user manual #

## Contents ##

* [Features](#features)
* [General usage](#general-usage)
* [Installation](#installation)
   * [Overl requirements](#overal-requirements)
   * [Windows\Linux\Mac](#windowslinuxmac)
   * [Jupyter kernel definition](#jupyter-kernel-definition)
* [Loading and preparing data](#loading-and-preparing-data)
  * [Input](#input)
  * [Output](#output)
  * [Tutorial](#tutorial)
* [Applications](#applications)
    * [Breastmilk microbiome](#breastmilk-rna-sequence-infant-gut-microbiome-and-metabolites-analysis)
    * [iHMP](#metagenomes-targeting-diverse-body-sites-in-multiple-time-points)
    * [CD4 counts](#treatment-effect-on-longitudinal-cd4-counts-)
    * [Inflammotory bowel disease](#identifying-important-metabolites-associated-with-inflammatory-bowel-disease-)
* [Support](#support)

------------------------------------------------------------------------------------------------------------------------------

# Features

1. Generic software that can handle any kind of sequencing data and phenotypes
2. One place to perform all analyses and produce high-quality visualizations
3. Optimized computation
4. User-friendly software
5. Provides temporal dynamics, associated omics features, and metadata
6. Enhanced with diagnostic and summarizing visualizations


# General usage

Running _waveome_ requires multiple steps, including installing _waveome_ package, loading data in the required format,
specifying covariates and outcomes (
omics features), running kernel services (takes some time and computing resources), and visualizing overall and
individual associations. All these steps are
demonstrated in the [waveome_overview.ipynb](https://github.com/omicsEye/waveome/blob/main/waveome_overview.ipynb)
notebook as a template (an example of the package modeling simulated data)
that you can use and modify for your input data. Each step is explained with details in following sections of this
tutorial.
# Installation
To install the package it is suggested that you create a new conda environment (required to have Python >= 3.9 and <= 3.11 for tensorflow)   

## Overall requirements
Installation of _waveome_ is processed in a conda environment. You therefore need to install _conda_ first. Go to the [Anaconda website](https://www.anaconda.com/) and download the latest version for your operating system.
* **For Windows users:** do not forget to add _conda_ to your system PATH. It will be asked as a part of installation procedure. 
* Make sure about _conda_ availability. Open a terminal (or command line for Windows users) and run:
```
conda --version
```

It should output something like:

```
conda 23.7.4
```

if not, you must make *conda* available to your system for further steps.
If you have problems adding conda to PATH, you can find instructions
[here](https://docs.anaconda.com/anaconda/user-guide/faq/).

## Windows\Linux\Mac
If you are using Windows operating system, please make sure you have both [git](https://gitforwindows.org/) and 'Microsoft Visual C++ 14.0' or later installed.
You need also to install [Microsoft C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
In case you face issues with this step, [this link](https://github.com/pycaret/pycaret/issues/1254) may help you.

Regardless of what your operating system is, follow these steps:

1. Open a terminal in your Linux or Mac system or command (`ctrl+R` then type `cmd` and press Enter) in your Windows system and use the following code to create a conda environment:
    ```commandline
    conda create --name waveome_env python=3.11
    ```
2. Activate your _conda_ environment:
    ```commandline
    conda activate waveome_env 
    ```
3. If you want to use _waveome_ in a Python notebook, for instance in Jupyter Notebook 
(which is recommended for running [waveome_overview.ipynb](https://github.com/omicsEye/waveome/blob/main/waveome_overview.ipynb) sample file 
and [example projects](https://github.com/omicsEye/waveome/tree/main/examples)), we recommend the installation of 
_Jupyter Notebook_ in this environment prior to the `pip` installation of _waveome_. To do so, if you are using any operating system **except Mac M1/M2**, 
simply run:
    ```commandline
    conda install jupyter 
    ```
    in the `waveome_env` in your terminal or command prompt and go to step 
[4](#item4). But, if **you are an M1/M2 Mac user**, <ins>prior to installation 
of _Jupyter Notebook_</ins> run the following in the `waveome_env`:
    ```commandline
    conda install -c conda-forge grpcio
    ```

    and afterwards run:

    ```commandline
    conda install jupyter
    ```
4. Install _waveome_ directly from GitHub:

    ```commandline
    python -m pip install git+https://github.com/omicsEye/waveome
    ```

## Jupyter kernel definition

To employ _waveome_ in _Jupyter Notebook_ we need to provide the kernel. This can be done with
```commandline
conda install ipykernel
```
and then
```commandline
python -m ipykernel install --user --name=waveome
```
in the terminal while `waveome_env` is active.



## Run using Jupyter Notebook
If you would like to run `waveome_overview.ipynb` then you should also set up a Jupyter kernel for the new waveome environment. This can be done with 
`conda install -n waveome ipykernel` and then 
`python -m ipykernel install --user --name=waveome`

Change directory to where you have your iPhyton notebook
`cd /PATH-TO_YOUR_iPythonNotebook-DiRECTORY`

Then run jupyter notebook in command line
`jupyter notebook`.

# Loading and preparing data

<!--# Getting Started with waveome

## Test waveome

To test if waveome is installed correctly, you may run the following command in the terminal:

```#!cmd
waveome -h
```

Which yields waveome command line options.

## Options ##

```

```
-->
## Input
As an input, _waveome_ requires a pandas data frame which contains at least: 
1. Subjects/individuals/patients index column,
2. Columns of covariates; in longitudinal studies these columns 
contain the time of observing the sample.
3. An Omics feature measurement.

All the above-mentioned items must be available in numeric types (`int` or `float`) in the data frame:
![sampledata](https://github.com/omicsEye/waveome/blob/main/figures/sample.png?raw=True)


The _Subject index_ is used to measure the subject effect. The categorical factors are encouraged to be considered 
through dummy variables. For instance, in the above example the factor `Sex` is considered as 'is the subject Female?' 
and '1' means "yes". _waveome_ does not consider the samples with missing values and it is required to delete 
the rows with missing values prior to `GPSearch`. 

## Output
The output of `GPSearch.run_serach` contains the results for each Bayesian nonparametric regression model 
fit on the data corresponding a kernel (or summation or multiplication of kernels) function including but not 
restricted to the BIC, corresponding parameter estimations and residuals. Based on information criterion, the 
best kernel is selected and the coefficients of determination of each omics feature and all the 
covariates can be displayed. The estimated mean function of the omics feature as a function of each covariate 
alongside the corresponding residual is provided. Depend on the response distribution assumption on the omics feature (Gaussian and 
Poisson for now; but the negative binomial distribution is also under construction) the posterior mean of the 
omics feature is also included as an output. We refer the users to see the outputs of 
[waveome_overview.ipynb](https://github.com/omicsEye/waveome/blob/main/waveome_overview.ipynb) ipython notebook 
file for more details.


<!--## Demo

```commandline
waveome -sf PATH_TO_SEQUENCE.FASTA -st aa -md PATH_TO_META_DATA.tsv -mv
 META_VARIABLE_NAME -a reg  -dth 0.15 --plot --write
```
-->

<!-- ß
### Running Kernel search

### Visualization
-->

## Tutorial

Multiple detailed ipython notebook of _waveome_ implementations are available in the
[examples](https://github.com/omicsEye/waveome/tree/master/examples) and the
required data for the examples are also available either in the
[data](https://github.com/omicsEye/waveome/tree/master/data) directory or the corresponding application directory.

# Applications

Here we try to use the _waveome_ on different datasets and elaborate on the results.

## Breastmilk RNA sequence, infant gut microbiome and metabolites analysis
[GWDBB](https://github.com/gwcbi/GWDBB/tree/master) is a reference data library for clinical trials and omics 
data. It contains the longitudinal gut microbiome and metabolomics data 
of infants and mothers breast milk RNA in different time-points. Two different 
longitudinal analysis has been derived on the data and can be found in 
[breastmilk_infant_metabolites_Poisson.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/Breastmilk/breastmilk_infant_metabolites_Poisson.ipynb) and 
[Breastmilk_infant_Microbiome.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/Breastmilk/Breastmilk_infant_Microbiome.ipynb) 
notebook files. 


## Metagenomes targeting diverse body sites in multiple time-points

[iHMP](https://www.nature.com/articles/nature23889) provided one of the broadest datasets for human microbiome 
data hosted in different niches in the body at different time-points. The available dataset has been collected out of 
265 individuals. The longitudinal analysis for different body sights are presented in 
[multioutput_ihmp.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/iHMP/multioutput_ihmp.ipynb). 

## Treatment effect on longitudinal CD4 counts 
The bivariate responses of HIV-1 RNA (count/ml) in seminal and blood of patients in HIV-RNA AIDS 
studies from Seattle, Swiss and UNCCH cohorts are considered in this example. The data were
collected out of N = 149 subjects divided into two groups of patients who were receiving a therapy
(14=106 patients) and those with no therapy or unknown therapy method (43 patients). The covariates
are scaled time, baseline age, baseline CD4 and two factors consists of group and cohort. Data are also 
available through Wang (2013). The analysis using _waveome_ is also provided in 
[CD4.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/CD4/CD4.ipynb).

Wang, W.-L. (2013), Multivariate t linear mixed models for irregularly observed multiple repeated measures 
with missing outcomes. _Biom. J._, **55**: 554-571. 
[10.1002/bimj.201200001](https://onlinelibrary.wiley.com/doi/10.1002/bimj.201200001)

## Identifying important metabolites associated with inflammatory bowel disease 

![ihmp](https://github.com/omicsEye/waveome/blob/main/figures/ihmp.png?raw=True)

We used metabolomics data from iHMP (Inflammatory Bowel Diseases)
project [Lloyd-Price et al. (2017)](https://doi.org/10.1038/s41586-019-1237-9) for this application. Our goal was to
characterize temporal dynamics of metabolites associated with severity of IBD and other patient characteristics.
This [Jupyter Notebook](https://github.com/omicsEye/waveome/blob/main/examples/iHMP/iHMP%20Data%20Overview.ipynb)
illustrates the steps.

<!-- 
## Novel insights of niche associations in the oral microbiome
-->

<!-- ![hmp](https://github.com/omicsEye/waveome/blob/master/img/hmp/hmp.png?raw=True) -->  
<!--Microbial species tend to adapt at the genome level to the niche in which they live. We hypothesize
that genes with essential functions change based on where microbial species live. Here we use microbial strain
representatives from stool metagenomics data of healthy adults from the
[Human Microbiome Project](https://doi.org/10.1038/nature11234). The input for _waveome_ consists of 1) an MSA file
with 1006 rows, each a representative strain of a specific microbial species, here Haemophilus parainfluenzae, with
49839 lengths; and 2) labels for waveome prediction are body sites from which samples were collected.
This [Jupyter Notebook](https://github.com/omicsEye/waveome/blob/master/examples/discrete_phenotype_HMP.ipynb)
illustrates the steps.


## Reveals important SARS-CoV-2 regions associated with Alpha and Delta variants


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
-->

<!-- <h2 id="hiv">
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
-->
# Support

* Please submit your questions or issues with the software at
  [Issues tracker](https://github.com/omicsEye/waveome/issues).
* For community discussions, questions, and issue reporting, please visit our
  forum [here](https://forum.omicseye.org/c/omics-downstream-analysis/waveome/12)