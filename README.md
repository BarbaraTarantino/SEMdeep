# SEMdeep
Structural Equation Modeling with Deep Neural Network and Machine Learning 

**SEMdeep** train and validate a custom (or data-driven) structural equation
    model (SEM) using layer-wise deep neural networks (DNNs) or node-wise machine
	learning (ML) algorithms. **SEMdeep** comes with the following functionalities:

- Automated ML or DL model training based on SEM network structures.

- Network plot representation as interpretation diagram.

- Model performance evaluation through regression and classification
  metrics.

- Model variable importance computation through Shapley (R2) values,
  Gradient (or Connection) weight approach and significance tests of
  network inputs.

## Installation

The latest stable version can be installed from CRAN:

``` r
install.packages("SEMdeep")
```

The latest development version can be installed from GitHub:

``` r
# install.packages("devtools")
devtools::install_github("BarbaraTarantino/SEMdeep")
```

