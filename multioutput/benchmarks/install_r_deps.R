# install_r_deps.R
# Reproducibility script for mogp-waveome R dependencies

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos='https://cloud.r-project.org')

# 1. Install Bioconductor core dependencies
bioc_pkgs <- c("edgeR", "fgsea", "impute", "pcaMethods", "MSnbase", "siggenes", "mixOmics", "timeOmics")
BiocManager::install(bioc_pkgs, update=FALSE, ask=FALSE)

# 2. Install CRAN helpers
cran_pkgs <- c("remotes", "Rcpp", "Rserve", "lme4", "igraph", "Cairo", "DEoptimR",
               "fastGHQuad", "robustbase", "lmerTest", "robustlmm", "dplyr")
install.packages(cran_pkgs, repos='https://cloud.r-project.org')

# 3. Install specialized GitHub packages
# MetaboAnalystR (requires 'qs' first)
remotes::install_github("traversc/qs")
remotes::install_github("xia-lab/MetaboAnalystR", build_vignettes=FALSE, upgrade=FALSE)

# PAL and its dependency PASI
remotes::install_github("elolab/PASI")
remotes::install_github("elolab/PAL")

# lmms: required by timeOmics method
remotes::install_github("cran/lmms", upgrade="never")

print("R dependencies installation complete.")
