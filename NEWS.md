## Version 1.1.0 Release Notes
* All new and revised torch functions no longer require the "cito" package. 

* Added new argument algo = c("nodewise","layerwise","structured","neuralgraph")
to the SEMdnn() function. Four algorithms are now implemented using R MLPs (number of
nodes with non-zero incoming connectivity) for "nodewise", L<R MLPs (number of layers
in the input graph) for "layerwise", and 1 MLP for "structured" or "neuralgraph".

* Delete algo="nn" and algo="dnn" in the SEMml() function. These NNs for a small
neural network model (1 hidden layer and 10 nodes) and large neural network model
(1 hidden layers and 1000 nodes) run with algo="nodewise" of the SEMdnn() function.

* Added new function getLOCO(). For SEMml() algo=c("sem","tree","rf","xgb"),
computes the contributions of each variable to individual predictions using LOCO
(Leave Out COvariates) values based on ghost variables. A CPU-efficient procedure. 

* Various fixed bugs discovered after the release 1.0.0.

## Version 1.0.0 Release Notes
* Version 1.0.0 is a major release with several new features, including:

* Added new argument outcome = NULL (defult). This parameter is used in SEMdnn()
and SEMml() functions to process a sink categorical node (as a factor) for
classification purposes using all graph nodes as covariates.

* Added new argument newoutcome = NULL (defult). This parameter is used in predict
(.SEM, .DNN, .ML) functions to predict a sink categorical node (as a factor) for
classification purposes using all graph nodes as covariates.

* classificationReport() function. A report showing the main classification metrics,
like precision, recall, F1-score, accuracy, Matthew's correlation coefficient (mcc)
for all classes of the node = as.factor(outcome).  

* crossValidation() function. A R-repeated K-fold cross-validation with a list of
M models from SEMrun(), SEMml() and SEMdnn(). The winning model is selected by reporting
the mean predicted performances across all RxKxM runs.

* getVariableImportance() function. Extraction of common Machine Learning (ML) variable
(predictor) importance measures after fitting SEMrun(), SEMml() or SEMdnn() models.

* Added new argument nboot = 0 (default). This parameter implements cheap bootstrapping
in SEMdnn() and SEMml() functions to generate uncertainties, i.e. CIs, for DNN/ML
parameters. Bootstrapping can be enabled by setting a small number (from 1 to 10) of
bootstrap samples.

* Change argument thr = 0.5 * max(abs(parameters)) (default). Now the DAG can be colored
using a numeric [0-1] threshold. For example, 1/0.5 = 2, can be interpreted as the number
of times a node/edge parameter is less than the maximum parameter value.

* Various fixed bugs discovered after the release 0.1.0.

## Version 0.1.0 Release Notes
* First stable version on CRAN
