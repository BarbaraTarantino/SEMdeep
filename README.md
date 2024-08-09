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


<p align="center">
  <img src="https://github.com/BarbaraTarantino/SEMdeep/blob/master/docs/figures/SEMdeep_workflow.png" width=100% height=100%>
</p>


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

**SEMdeep** uses the deep learning framework 'torch'. The **torch**  package
is native to R, so it's computationally efficient, as there is no need to install
Python or any other API. In order to install **torch** please follow these steps:

``` r
install.packages("torch")

library(torch)

install_torch(reinstall = TRUE)
```

If you have problems installing **torch** package, check out the
[installation](https://torch.mlverse.org/docs/articles/installation.html/)
help from the torch developer.

## Getting help

The full list of **SEMdeep** functions with examples is available at our website [**HERE**](https://barbaratarantino.github.io/SEMdeep/).

