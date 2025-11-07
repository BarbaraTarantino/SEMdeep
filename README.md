# SEMdeep
Structural Equation Modeling with Deep Neural Network and Machine Learning 

**SEMdeep** train and validate a custom (or data-driven) structural equation
    model (SEM) using deep neural networks (DNNs) or machine learning (ML)
	algorithms. **SEMdeep** comes with the following functionalities:

- Automated DNN or ML model training based on SEM network structures.

- Network plot representation as interpretation diagram.

- Model performance evaluation through regression and classification metrics.

- Compute model variable importance for a DNN (connection weights, gradient
  weights, or significance tests of network inputs) and for an ML (variable
  importance measures, Shapley (R2) values, or LOCO values).

## Installation

**SEMdeep** uses the deep learning framework 'torch'. The **torch** package
is native to R, so it's computationally efficient, as there is no need to install
Python or any other API, and DNNs can be trained on CPU, GPU and MacOS GPUs.
Before using 'SEMdeep' make sure that the current version of ‘torch’ is installed
and running: 

``` r
install.packages("torch")

library(torch)

install_torch(reinstall = TRUE)

```

Only for windows (not Linux or Mac). Some Windows distributions don’t have the
Visual C++ runtime pre-installed, download from
[Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170/)
**VC_redist.x86.exe** (R32) or **VC_redist.x86.exe** (R64) and install it.

For GPU setup, or if you have problems installing **torch** package, check out the
[installation](https://torch.mlverse.org/docs/articles/installation.html/)
help from the torch developer.

Then, the latest stable version can be installed from CRAN:

``` r
install.packages("SEMdeep")
```

The latest development version can be installed from GitHub:

``` r
# install.packages("devtools")
devtools::install_github("BarbaraTarantino/SEMdeep")
```

## Getting help

The full list of **SEMdeep** functions with examples is available at our website
[**HERE**](https://BarbaraTarantino.github.io/SEMdeep/).
