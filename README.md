# SEMdeep <img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/hex_SEMdeep.png" align="right" width="140"/>

[![CRAN status](https://www.r-pkg.org/badges/version/SEMdeep)](https://CRAN.R-project.org/package=SEMdeep)
[![R-CMD-check](https://github.com/BarbaraTarantino/SEMdeep/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/BarbaraTarantino/SEMdeep/actions)
[![pkgdown site](https://img.shields.io/badge/docs-pkgdown-blue.svg)](https://barbaratarantino.github.io/SEMdeep/)

---

## Overview

**SEMdeep: Structural Equation Modeling with Deep Neural Networks and Machine Learning**  
Authors: **Barbara Tarantino**, **Mario Grassi**  
Maintainer: *Barbara Tarantino (University of Pavia)*  
CRAN: [**SEMdeep**](https://CRAN.R-project.org/package=SEMdeep)  
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

## Modeling Framework

<img src="https://raw.githubusercontent.com/BarbaraTarantino/SEMdeep/main/man/figures/SEMdeep_slide3_algorithms.png" width="850"/>

---

### **Input**

**SEMdeep** operates on two main inputs:
- A **data matrix**, containing observed or simulated variables.  
- A **directed graph (adjacency matrix)** that encodes the hypothesized or data-driven causal relationships among variables.  

These inputs define the structural dependencies that guide model compilation, training, and explanation.

---

### **Modeling Strategies**

#### **SEMml()**

`SEMml()` performs *machine learning–based SEM fitting* through a **nodewise approach**.  
Each node with incoming edges is modeled as a function of its direct parents, resulting in a set of independent predictive models that follow the directed structure of the input graph.  
Supported algorithms include **SEM**, **tree**, **random forest**, and **XGBoost**, allowing flexible modeling of causal mechanisms at the node level.  
Bootstrapping can be applied to estimate confidence intervals for ML parameters (Lam, 2002).

#### **SEMdnn()**

`SEMdnn()` extends this logic to **deep neural networks (DNNs)**, supporting multiple graph-aware fitting strategies:

| **Mode** | **Fitting Strategy** | **Description** | **Reference** |
|-----------|----------------------|-----------------|----------------|
| **Nodewise** | Equation-by-equation | Fits one neural model per dependent variable using its direct parent nodes as predictors, following the causal edge structure. | Zheng et al. (2020) |
| **Layerwise** | Layer-by-layer | Trains sequential DNNs across graph layers, fitting each layer’s nodes as outputs of their upstream inputs. | Grassi & Tarantino (2025) |
| **Structured** | Whole-graph (masked) | Builds a Structured Neural Network with weight matrices masked to match the graph adjacency. | Chen et al. (2023) |
| **NeuralGraph** | Whole-graph (constrained) | Learns a Neural Graphical Model enforcing adjacency complement constraints for global causal consistency. | Shrivastava & Chajewska (2023) |

Each network is trained using **torch**, enabling GPU acceleration and efficient backpropagation.  
This design allows seamless transition from traditional SEMs to fully neural, graph-informed architectures.

---

### **Output and Explainability**

**SEMdeep** produces both predictive and interpretable outputs:

- **Model Estimates:** nodewise or layerwise fitted models with learned weights.  
- **Evaluation Metrics:** model performance indicators such as *R²*, *AMSE*, and *SRMR*.  
- **Explainability Measures:**  
  - `getConnectionWeight()`  
  - `getGradientWeight()`  
  - `getShapleyR2()`  
  - `getLOCO()`  

These functions provide interpretable representations of the fitted structure, highlighting the strength and direction of effects through color-coded causal graphs.  
Together, they make **SEMdeep** a bridge between predictive accuracy and causal transparency.

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

