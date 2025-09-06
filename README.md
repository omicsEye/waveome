# waveome

## Overview

<img style="float: right;" width="200" height="200" src="./figures/hex-waveome.png">

This repository houses code for the _waveome_ package - an easy to use and powerful Python library that analyzes longitudinal data using Gaussian processes. It is particularly well-suited to characterize the temporal dynamics of omics measurements and associated variables of interest. This is done by using the Gaussian process as a prior to allow for flexible, nonparametric estimation of the potential relationships between varibles of interest. Futhermore, we allow for automated variable selection through a variety of methods. The software is open source and is built on top of GPflow (and TensorFlow).

### Key features
* **General Purpose**: Focus for longitudinal data analysis, but also useful for cross-sectional hypotheses
* **Flexible Modeling**: Variety of kernels (including for categorical variables) and non-Gaussian likelihoods available
* **Variable Selection**: Search-based as well as global penalization with Horshoe priors to automatically identify relevant covariates and kernel structure
* **Metrics & Visualizations**: Generalized deviance explained and Bayes factors available as well as a variety of plotting features
* **Parallelization**: Independent model hyperparameter optimization occurs in parallel through [Ray](https://docs.ray.io) allowing scalability from local machine to clusters
<!-- * **Documentation**: Open-source GitHub repository of code complete with tutorials and a wide range of real-world applications. -->

## Installation

We recommend a fresh conda environment (Python 3.9–3.11):

```bash
conda create -n waveome_env python=3.11
conda activate waveome_env
pip install git+https://github.com/omicsEye/waveome
```

Recommended for Jupyter notebooks:
```bash
conda install jupyter ipykernel
python -m ipykernel install --user --name=waveome_env
```
For platform-specific tips, see `docs/INSTALL.md` (optional).

## Quick Start
```python
import seaborn as sns
from waveome.model_search import GPSearch

# Load example dataset
iris = sns.load_dataset("iris")

# Load waveome object
# Assume outcomes are sepal_length and sepal_width
gps = GPSearch(
  X=iris[["petal_length", "petal_width", "species"]],
  Y=iris[["sepal_length", "sepal_width"]],
  categorical_vars=["species"]
)

# Optimize GP models via penalization
gps.penalized_optimization()

# Visualize results
gps.plot_heatmap(var_cutoff=0, cluster=False)
```
See the tutorial notebook `waveome_overview.ipynb` for longitudinal synthetic data generation and more visualization options post-fitting.

## Applications
### Simulations:
Path: `examples/simulations/`\
Summary: We evaluated our methods on simulated data both for holdout distributional fit as well as our automated variable selection strategies. These were performed on the [GW HPC](https://it.gwu.edu/hpc-pegasus), but individuals might be interested in understanding more of the modeling components and methods in `waveome` which can be found in the notebook `simple_regression_different_models.ipynb`. 

### iHMP longitudinal metabolome:
Path: `examples/iHMP/`\
Summary: We used metabolomics data from iHMP (Inflammatory Bowel Disease) project [Lloyd-Price et al. (2017)](https://doi.org/10.1038/s41586-019-1237-9) for this application. Our goal was to characterize temporal dynamics of metabolites associated with severity of IBD while controlling for other patient/sample characteristics. The notebook `ihmp_waveome.ipynb` shows the analysis.

### Marine microbiome (In progress):
Path: `examples/Marine_microbiome/`\
Summary: We analyzed 28 observations of repeated microbiome samples taken in a marine environment pre and post treatment shock times. Our analysis focused on evaluating the relationship between the abundance of sequence variants and the treatment administered, while controlling for other environmental factors. The preliminary results can be seen in `16S_environment_microbiome_antibiotic_treatments.ipynb`.

### Breastmilk RNA and infant microbiome & metabolome (In progress):
Path: `examples/Breastmilk/`\
Summary: [GWDBB](https://github.com/gwcbi/GWDBB/tree/master) is a reference data library for clinical trials and omics 
data. One study contains the longitudinal gut microbiome and metabolomics data of infants and mothers breast milk RNA collected at multiple time points. Two longitudinal analyses have been performed and can be found in `breastmilk_infant_metabolites_Poisson.ipynb` and `Breastmilk_infant_Microbiome.ipynb` notebook files. 

### HIV CD4 counts (In progress):
Path: `examples/CD4/`\
Summary: The bivariate responses of HIV-1 RNA (count/ml) in seminal and blood of patients in HIV-RNA AIDS studies from Seattle, Swiss and UNCCH cohorts are considered in this example. The data were
collected out of N = 149 subjects divided into two groups of patients who were receiving a therapy (106 patients) and those with no therapy or unknown therapy method (43 patients). The covariates are scaled time, baseline age, baseline CD4 and two factors consists of group and cohort. Data are also 
available through [Wang (2013)](https://onlinelibrary.wiley.com/doi/10.1002/bimj.201200001). The analysis using `waveome` is provided in [CD4.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/CD4/CD4.ipynb).

---

<!-- ## Metagenomes targeting diverse body sites in multiple time-points

[iHMP](https://www.nature.com/articles/nature23889) provided one of the broadest datasets for human microbiome 
data hosted in different niches in the body at different time-points. The available dataset has been collected out of 
265 individuals. The longitudinal analysis for different body sights are presented in 
[multioutput_ihmp.ipynb](https://github.com/omicsEye/waveome/blob/main/examples/iHMP/multioutput_ihmp.ipynb).  -->

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

## Citation
If you use `waveome`, please cite:
> Allen Ross, Ali Reza Taheriouyn, Jason Llyod-Price, Ali Rahnavard (2024).
waveome: characterizing temporal dynamics of metabolites in longitudinal studies, https://github.com/omicsEye/waveome/.

## Support

* Issues: https://github.com/omicsEye/waveome/issues
* ~~Community forum: https://forum.omicseye.org/c/omics-downstream-analysis/waveome~~