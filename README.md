# SEMdeep <img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/hex_SEMdeep.png" align="right" width="140"/>

[![CRAN status](https://www.r-pkg.org/badges/version/SEMdeep)](https://CRAN.R-project.org/package=SEMdeep)
[![R-CMD-check](https://github.com/BarbaraTarantino/SEMdeep/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/BarbaraTarantino/SEMdeep/actions)
[![pkgdown site](https://img.shields.io/badge/docs-pkgdown-blue.svg)](https://barbaratarantino.github.io/SEMdeep/)

---

## Overview

**SEMdeep: Structural Equation Modeling with Deep Neural Networks and Machine Learning**  
Authors: **Barbara Tarantino**, **Mario Grassi**  
Maintainer: *Barbara Tarantino (University of Pavia)*  
CRAN: [https://CRAN.R-project.org/package=SEMdeep](https://CRAN.R-project.org/package=SEMdeep)  
Companion package: [**SEMgraph**](https://CRAN.R-project.org/package=SEMgraph)

---

## Description

**SEMdeep** provides an integrated framework to train, validate, and explain *Structural Equation Models (SEMs)* using **machine learning** and **deep neural network** algorithms.  
It combines the strengths of causal inference and predictive modeling to build interpretable systems that remain faithful to a given graph structure.

The package is fully compatible with **[SEMgraph](https://CRAN.R-project.org/package=SEMgraph)**, which defines and manipulates causal graphs.  
Together, the two packages form a dual modular system:

- **SEMgraph**: graph learning, model specification, and topological representation.  
- **SEMdeep**: SEM-based machine learning, deep learning fitting, evaluation, and explainability.

---

## Scientific Rationale and Added Value

<img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/SEMdeep_slide1_overview.png" width="820"/>

While most machine learning models achieve strong predictive performance, they often obscure the causal structure underlying the data.  
**SEMdeep** extends the classical SEM paradigm by embedding ML and DNN algorithms directly within the causal graph, thus providing:

- data-driven estimation aligned with prior causal structure;  
- model interpretability through connection and gradient weights;  
- interoperability with **SEMgraph** for graph construction and validation.

This integration allows the user to move seamlessly from causal discovery to explainable predictive modeling.

---

## Conceptual Workflow

<img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/SEMdeep_slide2_workflow.png" width="850"/>

The conceptual workflow follows a modular and interoperable design:

1. **Data preprocessing** – sampling, scaling, and feature selection.  
2. **Graph compilation** – conversion of DAGs or SEM specifications to adjacency matrices using `SEMgraph`.  
3. **Model training** – graph-aligned machine learning or deep neural network fitting via `SEMml()` and `SEMdnn()`.  
4. **Parameter optimization** – hyperparameter tuning and regularization across models.  
5. **Model evaluation** – predictive accuracy and goodness-of-fit metrics.  
6. **Model explanation** – quantitative and visual interpretation of causal effects.

All preprocessing and graph compilation steps are handled by **SEMgraph**, while model training, optimization, and explainability modules are managed internally by **SEMdeep**.

---

## Modeling Functions and Algorithms

<img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/SEMdeep_slide3_algorithms.png" width="850"/>

### SEMml()

`SEMml()` performs *machine learning–based SEM fitting* through a **nodewise approach**.  
Each node with incoming edges is modeled as a function of its direct parents, resulting in a set of independent predictive models that follow the directed structure of the input graph.  
Supported algorithms include **SEM**, **tree**, **random forest**, and **XGBoost**.  
An optional lightweight bootstrap (Lam, 2002) can be used to estimate uncertainty around parameters.

### SEMdnn()

`SEMdnn()` extends this logic to deep neural networks, supporting multiple fitting strategies:

| Mode | Description | Reference |
|------|--------------|-----------|
| **Nodewise** | Fits one neural model per dependent variable | Zheng et al. (2020) |
| **Layerwise** | Learns topological layers from sink to source nodes | Grassi & Tarantino (2025) |
| **Structured** | Constrains weight matrices to reflect graph connectivity (Structured NN) | Chen et al. (2023) |
| **NeuralGraph** | Trains a Neural Graphical Model under adjacency complement constraints | Shrivastava & Chajewska (2023) |

Each network is trained using **torch**, enabling GPU acceleration and efficient backpropagation.  
Bootstrapping can be applied to estimate confidence intervals for neural parameters.  
This design allows the user to transition smoothly from classical SEMs to graph-aware deep learning.

---

## Model Input, Output, and Explainability

**Input:**  
A data matrix and a directed graph (adjacency matrix) defining causal relationships between variables.  

**Output:**  
- Nodewise or layerwise fitted models.  
- Parameter estimates or learned weights.  
- Model evaluation metrics (R², AMSE, SRMR).  
- Explainability measures (`getConnectionWeight()`, `getGradientWeight()`, `getShapleyR2()`, `getLOCO()`).

Model explanation functions provide interpretable visualizations of the fitted structure, coloring edges by effect strength and direction.

---

## Integration and Applications

<img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/SEMdeep_slide4_summary.png" width="800"/>

| Principle | Description |
|------------|-------------|
| **Integration** | Combines SEM, ML, and DL within a unified framework. |
| **Alignment** | Aligns model training with the causal topology of the input graph. |
| **Explanation** | Produces quantitative and graphical interpretation of causal effects. |
| **Application** | Suitable for biomedical, genomic, and multi-omics studies requiring causal insight. |

Together, **SEMgraph** and **SEMdeep** provide a coherent workflow for *Causal AI*, connecting structural modeling with modern predictive algorithms.

---

## Installation

`SEMdeep` is built on the native R framework **torch**, which does not require Python.  
It supports CPU, GPU, and Apple Silicon acceleration.

```r
# Install torch
install.packages("torch")
library(torch)
install_torch(reinstall = TRUE)

# Install SEMdeep
install.packages("SEMdeep")

# Development version
remotes::install_github("BarbaraTarantino/SEMdeep")
```
---

## Documentation

Comprehensive documentation, tutorials, and examples are available at:  
[https://barbaratarantino.github.io/SEMdeep/](https://barbaratarantino.github.io/SEMdeep/)

Related package:  
- **SEMgraph** — [CRAN](https://CRAN.R-project.org/package=SEMgraph) | [GitHub](https://github.com/BarbaraTarantino/SEMgraph)

---

## Contact

**Barbara Tarantino**  
Department of Brain and Behavioural Sciences, University of Pavia  
Email: [barbara.tarantino00@gmail.com](mailto:barbara.tarantino00@gmail.com)  
GitHub: [https://github.com/BarbaraTarantino](https://github.com/BarbaraTarantino)

