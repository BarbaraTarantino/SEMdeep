#  SEMdeep library
#  Copyright (C) 2024 Mario Grassi; Barbara Tarantino 
#  e-mail: <mario.grassi@unipv.it>
#  University of Pavia, Department of Brain and Behavioral Sciences
#  Via Bassi 21, 27100 Pavia, Italy

#  SEMdeep is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  SEMgraph is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# -------------------------------------------------------------------- #

#' @title Layer-wise SEM train with a Deep Neural Netwok (DNN)
#'
#' @description The function builds the topological layer (TL) ordering
#' of the input graph to fit a series of Deep Neural Networks (DNN) 
#' models, where the nodes in one layer act as response variables (output) 
#' y and the nodes in the sucessive layers act as predictors (input) x. 
#' Each fit uses the \code{\link[cito]{dnn}} function of the \pkg{cito} R
#' package, based on the deep learning framework 'torch'.
#'
#' The \pkg{torch} package is native to R, so it's computationally efficient
#' and the installation is very simple, as there is no need to install Python
#' or any other API, and DNNs can be trained on CPU, GPU and MacOS GPUs.
#' In order to install \pkg{torch} please follow these steps:
#'
#' \code{install.packages("torch")}
#'
#' \code{library(torch)}
#'
#' \code{install_torch(reinstall = TRUE)}
#'
#' For setup GPU or if you have problems installing \pkg{torch} package, check out the
#' \href{https://torch.mlverse.org/docs/articles/installation.html/}{installation}
#' help from the torch developer.
#'
#' @param graph An igraph object.
#' @param data A matrix with rows corresponding to subjects, and columns
#' to graph nodes (variables).
#' @param train A numeric vector specifying the row indices corresponding to 
#' the train dataset.
#' @param cowt A logical value. If \code{cowt=TRUE} connection weights of the
#' input node (variables) are computing (default = FALSE).
#' @param thr A numerical value indicating the threshold to apply on the absolute
#' values of the connection matrix to color the graph (default = NULL).
#' @param loss A character value specifying the loss after which network should 
#' be optimized. The user can specify: (a) "mse" (mean squared error), "mae" 
#' (mean absolute error), or "gaussian" (normal likelihood), for regression problems; 
#' (b) "poisson" (poisson likelihood), or "nbinom" (negative binomial likelihood)
#' for regression with count data; (c) "binomial" (binomial likelihood) for 
#' binary classification problems; (d) "softmax" or "cross-entropy" for multi-class
#' classification (default = "mse").
#' @param hidden hidden units in layers; the number of layers corresponds with
#' the length of the hidden units. As a default, \code{hidden = c(10L, 10L, 10L)}.
#' @param link A character value describing the activation function to use, which 
#' might be a single length or be a vector with many activation functions assigned
#' to each layer (default = "relu"). 
#' @param validation A numerical value indicating the proportion of the data set 
#' that should be used as a validation set (randomly selected, default = 0). 
#' @param bias A logical vector, indicating whether to employ biases in the layers 
#' (\code{bias = TRUE}), which can be either vectors of logicals for each layer 
#' (number of hidden layers + 1 (final layer)) or of length one (default = TRUE). 
#' @param lambda A numerical value indicating the strength of regularization: 
#' lambda penalty, \eqn{\lambda * (L1 + L2)} (default = 0). 
#' @param alpha A numerical vector to add L1/L2 regularization into the training. 
#' Set the alpha parameter for each layer to \eqn{(1 - \alpha) * ||weights||_1 + 
#' \alpha ||weights||^2}. It must fall between 0 and 1 (default = 0.5). 
#' @param dropout A numerical value for the dropout rate, which is the probability
#' that a node will be excluded from training (default = 0). 
#' @param optimizer A character value indicating the optimizer to use for 
#' training the network. The user can specify: "adam" (ADAM algorithm), "adagrad"
#' (adaptive gradient algorithm), "rmsprop" (root mean squared propagation),
#' "rprop” (resilient backpropagation), "sgd" (stochastic gradient descent).
#' As a default, \code{optimizer = “adam”}.
#' @param lr A numerical value indicating the learning rate given to the optimizer 
#' (default = 0.01). 
#' @param epochs A numerical value indicating the epochs during which the training 
#' is conducted (default = 100). 
#' @param device A character value describing the CPU/GPU device ("cpu", "cuda", "mps")
#' on which the  neural network should be trained on (default = "cpu") 
#' @param early_stopping If set to integer, training will terminate if the loss 
#' increases over a predetermined number of consecutive epochs and apply validation
#' loss when available. Default is FALSE, no early stopping is applied. 
#' @param verbose If \code{verbose = TRUE}, the training curves of the DNN models
#' are displayed as output, comparing the training, validation and baseline 
#' curves in terms of loss (y) against the number of epochs (x) (default = TRUE). 
#' @param ... Currently ignored.
#'
#' @details By mapping data onto the input graph, \code{SEMdnn()} creates a set 
#' of DNN models based on the topological layer (j=1,…,L) structure of the input
#' graph. In each iteration, the response (output) variables, y are the nodes in
#' the j=1,...,(L-1) layer and the predictor (input) variables, x are the nodes
#' belonging to the successive, (j+1),...,L layers. 
#' Each DNN model is a Multilayer Perceptron (MLP) network, where every neuron node
#' is connected to every other neuron node in the hidden layer above and every other
#' hidden layer below. Each neuron's value is determined by calculating a weighted
#' summation of its outputs from the hidden layer before it, and then applying an
#' activation function.  The calculated value of every neuron is used as the input
#' for the neurons in the layer below it, until the output layer is reached.
#'
#' @return An S3 object of class "DNN" is returned. It is a list of 5 objects:
#' \enumerate{
#' \item "fit", a list of DNN model objects, including: the estimated covariance 
#' matrix (Sigma), the estimated model errors (Psi), the fitting indices (fitIdx),
#' and the estimated connection weights (parameterEstimates), if cowt=TRUE.
#' \item "Yhat", the matrix of continuous predicted values of graph nodes  
#' (excluding source nodes) based on training samples. 
#' \item "model", a list of all j=1,...,(L-1) fitted MLP network models.
#' \item "graph", the induced DAG of the input graph mapped on data variables.
#' If cowt=TRUE, the DAG is colored based on the estimated connection weights, if
#' abs(W) > thr and W < 0,  the edge is inhibited and it is highlighted in blue;
#' otherwise, if abs(W) > thr and W > 0, the edge is activated and it is highlighted
#' in red. 
#' \item "data", training data subset mapping graph nodes.
#' }
#'
#' @import SEMgraph
#' @import igraph
#' @importFrom graphics par polygon
#' @importFrom mgcv gam
#' @importFrom stats coef coefficients cor density fitted formula lm predict p.adjust quantile reshape sd
#' @importFrom utils setTxtProgressBar tail txtProgressBar
#' @importFrom torch torch_is_installed    
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' Amesöder, C., Hartig, F. and Pichler, M. (2024), ‘cito': an R package for training
#' neural networks using ‘torch'. Ecography, 2024: e07143. https://doi.org/10.1111/ecog.07143
#' 
#' Grassi M, Palluzzi F, Tarantino B (2022). SEMgraph: An R Package for Causal Network
#' Analysis of High-Throughput Data with Structural Equation Models.
#' Bioinformatics, 38 (20), 4829–4830 <https://doi.org/10.1093/bioinformatics/btac567>
#'
#' @examples
#' 
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' group<- alsData$group
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' start<- Sys.time()
#' dnn0 <- SEMdnn(ig, data, train, cowt = TRUE, thr = NULL,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#' 
#' #str(dnn0, max.level=2)
#' dnn0$fit$fitIdx
#' dnn0$fit$parameterEstimates
#' gplot(dnn0$graph)
#' table(E(dnn0$graph)$color)
#' 
#' #...with a binary outcome (1=case, 0=control)
#' 
#' ig1<- mapGraph(ig, type="outcome"); gplot(ig1)
#' outcome<- ifelse(group == 0, -1, 1); table(outcome)
#' data1<- cbind(outcome, data); data1[1:5,1:5]
#' 
#' start<- Sys.time()
#' dnn1 <- SEMdnn(ig1, data1, train, cowt = TRUE, thr = NULL,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#' 
#' #str(dnn1, max.level=2)
#' dnn1$fit$fitIdx
#' dnn1$fit$parameterEstimates
#' gplot(dnn1$graph) 
#' table(E(dnn1$graph)$color)
#'
#' #...with input -> hidden structure -> output :
#' # source nodes -> graph layer structure -> sink nodes
#'
#' #Topological layer (TL) ordering
#' K<- c(12,  5,  3,  2,  1,  8)
#' K<- rev(K[-c(1,length(K))]);K
#' 
#' ig2<- mapGraph(ig, type="source"); gplot(ig2)
#'
#' start<- Sys.time()
#' dnn2 <- SEMdnn(ig2, data, train, cowt = TRUE, thr = NULL,
#' 			loss = "mse", hidden = 5*K, link = "selu",
#' 			#loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#'
#' #Visualization of the neural network structure
#' nplot(dnn2$model[[1]], bias=FALSE)
#'
#' #str(dnn2, max.level=2)
#' dnn2$fit$fitIdx
#' mean(dnn2$fit$Psi)
#' dnn2$fit$parameterEstimates
#' gplot(dnn2$graph)
#' table(E(dnn2$graph)$color)
#' }
#' }
#'
#' @export
#'

SEMdnn <- function(graph, data, train = NULL, cowt = FALSE, thr = NULL, 
	loss = "mse", hidden = c(10L, 10L, 10L), link = "relu", 
	validation = 0, bias = TRUE, lambda = 0, alpha = 0.5, dropout = 0,
	optimizer = "adam", lr = 0.01, epochs = 100, device = "cpu",
	early_stopping = FALSE, verbose = TRUE, ...)
{
	# Set graph and data objects :
	nodes <- colnames(data)[colnames(data) %in% V(graph)$name]
	graph <- induced_subgraph(graph, vids=which(V(graph)$name %in% nodes))
	dag <- graph2dag(graph, data, bap=FALSE) #del cycles & all <->
	din <- igraph::degree(dag, mode= "in")
	Vx <- V(dag)$name[din == 0]
	Vy <- V(dag)$name[din != 0]
	px <- length(Vx)
	py <- length(Vy)
		
	X <- data[, V(dag)$name]
	if (!is.null(train)) {
	  Z_train <- scale(X[train, ])
	  mp <- apply(X[train, ], 2, mean)
	  sp <- apply(X[train, ], 2, sd)
	  Z_test <- scale(X[-train, ], center=mp, scale=sp)
	}else{
	  train<- 1:nrow(X)
	  Z_train <- Z_test <- scale(X)
	}
	n <- nrow(Z_train)

	# Check whether a compatible GPU is available for computation.
	#use_cuda <- torch::cuda_is_available()
	#device <- ifelse(use_cuda, "cuda", "cpu")
	res <- parameterEstimates.DNN(dag, Z_train, Z_test,
	        loss, hidden, link, validation, bias, lambda,
			alpha, dropout, optimizer, lr, epochs, device,
			early_stopping, verbose)
	#str(res, max.level=1)

	# Get connection weights:
	if (cowt) {
	  dnn <- list(model=res[[1]], graph=dag)
	  cowt <- getConnectionWeight(object=dnn, thr=thr)
	  est <- list(beta=cowt$est, psi=res$sigma)
	  dag <- cowt$dag
	}else{
	  est <- list(beta=NULL, psi=res$sigma)
	  dag <- dag
	}

	# Shat and Sobs matrices :
	#Shat <- cor(cbind(Z_test[,Vx], res$yhat[,Vy]))
	#Sobs <- cor(Z_test[, c(Vx,Vy)])
	Shat <- cor(cbind(Z_train[,Vx], res$YHAT[,Vy]))
	Sobs <- cor(Z_train[, c(Vx,Vy)])
	E <- Sobs - Shat # diag(E)

	# Fit indices :
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	logL <- -0.5 * (sum(log(res$sigma)) + py * log(n))#GOLEM, NeurIPS 2020
	AMSE <- mean(res$sigma) #average Mean Square Error
	idx <- c(logL = logL, amse = AMSE, rmse = sqrt(AMSE), srmr = SRMR)
	it <- py * epochs
	message("\n", "DNN solver ended normally after ", it, " iterations")
	message("\n", " logL:", round(idx[1],6), "  srmr:", round(idx[4],6))
	
	fit <- list(Sigma=Shat, Beta=NULL, Psi=res[[2]], fitIdx=idx, parameterEstimates=est)
	res <- list(fit=fit, Yhat=res[[3]], model=res[[1]], graph=dag, data=X[train, ])
	class(res) <- "DNN"

	return(res)	
}

parameterEstimates.DNN <- function(dag, Z_train, Z_test,
	loss, hidden, link, validation, bias, lambda, alpha, dropout,
	optimizer, lr, epochs, device, early_stopping, verbose, ...)
{
	# Set objects:
	V(dag)$name<- paste0("z", V(dag)$name)
	colnames(Z_train)<- paste0("z", colnames(Z_train))
	colnames(Z_test)<- paste0("z", colnames(Z_test))
	L<- buildLevels(dag)
	#pe<- igraph::as_data_frame(dag)[,c(1,2)]
	#y<- split(pe, f=pe$to)
	#y; length(y); names(y)
	sigma<- NULL
	yhat<- NULL
	YHAT<- NULL
	est<- NULL
	dnn<- list()
		
	for (l in 1:(length(L)-1)) {
	  #cat((l, ":", L[[l]],"\n")
	  message(cat(l, ":", L[[l]]))
	  yn<- L[[l]]
	  xn<- unlist(L[(l+1):length(L)])
	  X<- data.frame(Z_train[,xn])
	  if (ncol(X) == 1) colnames(X)<- xn
	  Y<- data.frame(Z_train[,yn])
	  if (ncol(Y) == 1) colnames(Y)<- yn
	  
	  #fitting a dnn model to predict Y on X
	  nn.fit <- cito::dnn(data = Z_train, 
					loss = loss,
					hidden = hidden,
					activation = link,
					validation = validation,
					bias = bias, 
					lambda = lambda,
					alpha = alpha,
					dropout = dropout,
					optimizer = optimizer,
					lr = lr,
					epochs = epochs,
					plot = verbose,
					verbose = FALSE,
					device = device,
					early_stopping = early_stopping,
					X = X,
					Y = Y)

	  #results of the last iteraction
	  if (verbose) {
	   print(nn.fit$losses[epochs,])
	   message()
	  }
	  #TRAIN predictions and prediction error (MSE)
	  #pred <- predict(nn.fit, Z_train)
	  #pe <- apply((Z_test[,yn] - pred)^2, 2, mean)
	  #yhat <- cbind(yhat, pred)
	  PRED <- predict(nn.fit, Z_train)
	  pe <- apply((Z_train[,yn] - PRED)^2, 2, mean)
	  sigma <- c(sigma, pe)
	  dnn <- c(dnn, list(nn.fit))
	  YHAT <- cbind(YHAT, PRED)
	}
	
	#colnames(yhat) <- sub(".", "", unlist(L[-length(L)]))
	colnames(YHAT) <- sub(".", "", unlist(L[-length(L)]))
	names(sigma) <- sub(".", "", unlist(L[-length(L)]))
	
	return(list(dnn = dnn, sigma = sigma, YHAT = YHAT, est = est))
}

#' @title SEM-based out-of-sample prediction using layer-wise DNN
#'
#' @description Predict method for DNN objects.
#'
#' @param object A model fitting object from \code{SEMdnn()} function. 
#' @param newdata A matrix containing new data with rows corresponding to
#' subjects, and columns to variables.
#' @param verbose Print predicted out-of-sample MSE values (default = FALSE).
#' @param ... Currently ignored.
#'
#' @return A list of 2 objects:
#' \enumerate{
#' \item "PE", vector of the prediction error equal to the Mean Squared Error
#' (MSE) for each out-of-bag prediction. The first value of PE is the AMSE,
#' where we average over all (sink and mediators) graph nodes.
#' \item "Yhat", the matrix of continuous predicted values of graph nodes  
#' (excluding source nodes) based on out-of-bag samples. 
#' }
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # Load Amyotrophic Lateral Sclerosis (ALS)
#' data<- alsData$exprs; dim(data)
#' data<- transformData(data)$data
#' ig<- alsData$graph; gplot(ig)
#' group<- alsData$group 
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' start<- Sys.time()
#' dnn0 <- SEMdnn(ig, data, train, cowt = FALSE, thr = NULL,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#' mse0 <- predict(dnn0, data[-train, ], verbose=TRUE)
#' 
#' # SEMrun vs. SEMdnn MSE comparison
#' sem0 <- SEMrun(ig, data[train, ], SE="none", limit=1000)
#' mse0 <- predict(sem0, data[-train,], verbose=TRUE)
#' 
#' #...with a binary outcome (1=case, 0=control)
#' 
#' ig1<- mapGraph(ig, type="outcome"); gplot(ig1)
#' outcome<- ifelse(group == 0, -1, 1); table(outcome)
#' data1<- cbind(outcome, data); data1[1:5,1:5]
#' 
#' start<- Sys.time()
#' dnn1 <- SEMdnn(ig1, data1, train, cowt = TRUE, thr = NULL,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#'
#' mse1 <- predict(dnn1, data1[-train, ])
#' yobs <- group[-train]
#' yhat <- mse1$Yhat[ ,"outcome"]
#' benchmark(yobs, yhat, thr=0, F1=FALSE)
#' }
#' }
#'
#' @method predict DNN
#' @export
#' @export predict.DNN
#' 

predict.DNN <- function(object, newdata, verbose=FALSE, ...)
{
	dnn.fit <- object$model
	stopifnot(inherits(dnn.fit[[1]], "citodnn"))
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))

	yhat <- NULL
	yn <- c()
	for (j in 1:length(dnn.fit)) {
	  fit <- dnn.fit[[j]]
	  vn <- all.vars(fit$old_formula)
	  vx <- colnames(fit$data$X)
	  vy <- vn[vn %in% vx == FALSE]
	  pred <- predict(fit, Z_test)
	  yhat <- cbind(yhat, pred)
	  yn <- c(yn, vy)
	}

	yobs<- Z_test[, yn]
	PE<- colMeans((yobs - yhat)^2)
	pe<- mean(PE)
	colnames(yhat) <- sub(".", "", yn)
	names(PE) <- colnames(yhat)
	if (verbose) print(c(amse=pe,PE))
	
	return(list(PE=c(amse=pe,PE), Yhat=yhat))
}

#' @title Gradient Weight Approach for neural network variable importance
#'
#' @description  The function computes the gradient matrix, i.e., the average
#' conditional effects of the input variables w.r.t the neural network model,
#' as discussed by Amesöder et al (2024).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A threshold value to apply to gradient weights of input
#' nodes (variables). If NULL (default), the threshold is set to
#' thr=mean(abs(gradient weights)).
#' @param verbose A logical value. If FALSE (default), the processed graph
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details The partial derivatives method calculates the derivative (the gradient)
#' of each output variable (y) with respect to each input variable (x) evaluated at
#' each observation (i=1,...,n) of the training data. The contribution of each input 
#' is evaluated in terms of both magnitude taking into account not only the connection
#' weights and activation functions, but also the values of each observation of the
#' input variables. 
#' Once the gradients for each variable and observation, a summary gradient is calculated
#' by averaging over the observation units. Finally, the average weights are entered into
#' a matrix, W(pxp) and the element-wise product with the binary (1,0) adjacency matrix,
#' A(pxp) of the input DAG, W*A maps the weights on the DAG edges.
#' Note that the operations required to compute partial derivatives are time consuming
#' compared to other methods such as Olden's (connection weight). The computational
#' time increases with the size of the neural network or the size of the data. Therefore,
#' the function uses a progress bar to check the progress of the gradient evaluation per
#' observation.
#
#' @return A list od two object: (i) a data.frame including the connections together
#' with their weights, and (ii) the DAG with colored edges. If abs(W) > thr and W < 0,
#' the edge is inhibited and it is highlighted in blue; otherwise, if abs(W) > thr and
#' W > 0, the edge is activated and it is highlighted in red. 
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' Amesöder, C., Hartig, F. and Pichler, M. (2024), ‘cito': an R package for training
#' neural networks using ‘torch'. Ecography, 2024: e07143. https://doi.org/10.1111/ecog.07143
#' 
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#'
#' dnn0 <- SEMdnn(ig, data, train=1:nrow(data), cowt = FALSE,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' 
#' res<- getGradientWeight(dnn0, thr=NULL, verbose=TRUE)
#' table(E(res$dag)$color)
#' }
#' }
#'
#' @export
#' 

getGradientWeight<- function(object, thr = NULL, verbose = FALSE, ...)
{
	model <- object$model
	graph <- object$graph
	A <- as_adjacency_matrix(graph, sparse=FALSE)
	rownames(A) <- paste0("z", rownames(A))
	colnames(A) <- rownames(A)
	est <- NULL
	
	pb <- txtProgressBar(min = 0, max = length(model), style = 3)
	for (j in 1:length(model)) {
	  vn <- all.vars(model[[j]]$old_formula)
	  vx <- colnames(model[[j]]$data$X)
	  vy <- vn[vn %in% vx == FALSE]
	  if (length(vx) > 1) {
		 W <- quiet(print(cito::conditionalEffects(model[[j]])))
	  }else{
		 W <- matrix(0.5, nrow=1, ncol=length(vy))
	  }
	  WA <- as.matrix(W * A[vx, vy])
	  X <- gsub("z", "", vx)
	  Y <- gsub("z", "", vy)
	  
	  for (k in 1:length(Y)) {
	   label <- data.frame(
				 lhs = rep(Y[k], length(X)),
				 op = "~",
				 rhs = X)
	   est <- rbind(est, cbind(label, WA[,k]))
	  }
	  setTxtProgressBar(pb, j)
	}
	close(pb)
	est<- est[est[,4] != 0, ]
	rownames(est)<- NULL
	colnames(est)[4]<- "grad"
	class(est)<- c("lavaan.data.frame","data.frame")

	df<- data.frame(est[,3],est[,1],weight=est[,4])
	dag<- graph_from_data_frame(df)
	if (is.null(thr)) thr <- mean(abs(E(dag)$weight))
	dag<- colorDAG(dag, thr=thr, verbose=verbose)

	return(list(est = est, dag = dag))
}

colorDAG <- function(dag, psi=NULL, thr=NULL, verbose=FALSE, ...)
{
	# set node and edge colors :
	din<- igraph::degree(dag, mode= "in")
	dout<- igraph::degree(dag, mode = "out")
	#V(dag)$color[din == 0]<- "cyan"
	#V(dag)$color[dout == 0]<- "orange"
	if (!is.null(psi)) {
	 Ve <- names(psi)[psi < mean(psi)]
	 V(dag)$color <- ifelse(V(dag)$name %in% Ve, "pink", "white")
	 V(dag)$color[din == 0] <- "cyan"
	}
	enames <- attr(E(dag), "vnames")
	weight <- E(dag)$weight
	Er <- enames[abs(weight) > thr & weight < 0]
	Ea <- enames[abs(weight) > thr & weight > 0]
	E(dag)$color <- ifelse(attr(E(dag), "vnames") %in% 
			Er, "royalblue3", ifelse(attr(E(dag), "vnames") %in%
			Ea, "red2", "gray50"))
	E(dag)$width <- ifelse(E(dag)$color == "gray50", 1, 2)

	if (verbose) gplot(dag)

	return(dag)
}

#' @title Connection Weight Approach for neural network variable importance
#'
#' @description The function computes the product of the raw input-hidden and
#' hidden-output connection weights between each input and output neuron and
#' sums the products across all hidden neurons, as proposed by Olden (2004).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A value of the threshold to apply to connection weights. If NULL
#' (default), the threshold is set to thr=mean(abs(connection weights)).
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details In a neural network, the connections between inputs and outputs are 
#' represented by the connection weights between the neurons. 
#' The importance values assigned to each input variable using the Olden method are
#' in units that are based directly on the summed product of the connection weights.
#' The amount and direction of the link weights largely determine the proportional
#' contributions of the input variables to the neural network's prediction output.
#' Input variables with larger connection weights indicate higher intensities
#' of signal transfer and are therefore more important in the prediction process.
#' Positive connection weights represent excitatory effects on neurons (raising the
#' intensity of the incoming signal) and increase the value of the predicted response, 
#' while negative connection weights represent inhibitory effects on neurons 
#' (reducing the intensity of the incoming signal). The weights that change sign
#' (e.g., positive to negative) between the input-hidden to hidden-output layers
#' would have a cancelling effect, and vice versa weights with the same sign would
#' have a synergistic effect.
#' Note that in order to map the connection weights to the DAG edges, the element-wise
#' product, W*A is performed between the Olden's weights entered in a matrix, W(pxp)
#' and the binary (1,0) adjacency matrix, A(pxp) of the input DAG. 
#'
#' @return A list od two object: (i) a data.frame including the connections together
#' with their weights, and (ii) the DAG with colored edges. If abs(W) > thr and W < 0,
#' the edge is inhibited and it is highlighted in blue; otherwise, if abs(W) > thr and
#' W > 0, the edge is activated and it is highlighted in red. 
#'
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' Olden, Julian & Jackson, Donald. (2002). Illuminating the "black box": A 
#' randomization approach for understanding variable contributions in artificial
#' neural networks. Ecological Modelling. 154. 135-150. 10.1016/S0304-3800(02)00064-9. 
#' 
#' Olden, Julian. (2004). An accurate comparison of methods for quantifying 
#' variable importance in artificial neural networks using simulated data. 
#' Ecological Modelling. 178. 10.1016/S0304-3800(04)00156-5. 
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' 
#' dnn0 <- SEMdnn(ig, data, train=1:nrow(data), cowt = FALSE,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' 
#' res<- getConnectionWeight(dnn0, thr=NULL, verbose=TRUE)
#' table(E(res$dag)$color)
#' }
#' }
#'
#' @export
#' 

getConnectionWeight<- function(object, thr = NULL, verbose = FALSE, ...)
{
	model <- object$model
	graph <- object$graph
	A <- as_adjacency_matrix(graph, sparse=FALSE)
	rownames(A) <- paste0("z", rownames(A))
	colnames(A) <- rownames(A)
	est <- NULL
	
	for (j in 1:length(model)) {
	  W <- getWeight(model[[j]], A)
	  X <- gsub("z", "", rownames(W))
	  Y <- gsub("z", "", colnames(W))
	  
	  for (k in 1:ncol(W)) {
	   label<- data.frame(
				lhs = rep(Y[k], length(X)),
				op = "~",
				rhs = X)
	   est <- rbind(est, cbind(label, W[,k]))
	  }
	}
	est<- est[est[,4] !=  0, ]
	rownames(est)<- NULL
	colnames(est)[4]<- "weight"
	class(est)<- c("lavaan.data.frame","data.frame")
	
	df<- data.frame(est[,3],est[,1],weight=est[,4])
	dag<- graph_from_data_frame(df)
	if (is.null(thr)) thr <- mean(abs(E(dag)$weight))
	dag<- colorDAG(dag, thr=thr, verbose=verbose)
	
	return(list(est = est, dag = dag))
}

getWeight<- function(nn.fit, A, ...)
{
	# Matrix product of input-hidden, hidden-output weights,
	# proposed by Olden et al. Ecol. Model. 2004;389–397
	W <- coef(nn.fit)[[1]]
	w <- t(W[[1]])
	for (k in seq(3, length(W), by=2)) { 
	 w <- w %*% t(W[[k]])
	 if (length(W) == 4) break
	}
	
	vn <- all.vars(nn.fit$old_formula)
	vx <- colnames(nn.fit$data$X)
	vy <- vn[vn %in% vx == FALSE]
	wa <- as.matrix(w * A[vx, vy])
	rownames(wa) <- vx
	colnames(wa) <- vy
	
	return(wa)
}

#' @title Test for the significance of neural network inputs
#'
#' @description The function computes a formal test for the significance of
#' neural network input nodes, based on a linear relationship between the
#' observed output and the predicted values of an input variable, when all
#' other input variables are maintained at their mean values, as proposed by
#' Mohammadi (2018).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A value of the threshold to apply to input p-values. If thr=NULL
#' (default), the threshold is set to thr=0.05.
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details A neural network with an arbitrary architecture is trained, 
#' taking into account factors like the number of neurons, hidden layers, 
#' and activation function. Then, network's output is simulated to get 
#' the predicted values of the output variable, fixing all the inputs
#' (with the exception of one nonconstant input variable) at their mean
#' values; network’s predictions are saved, after doing this for each input
#' variable. As last step, multiple regression analysis is applied node-wise
#' (mapping the input DAG) on the observed output nodes with the predicted
#' values of the input nodes as explanatory variables. The statistical
#' significance of the coefficients is evaluated with the standard t-student
#' critical values, which represent the importance of the input variables. 
#' 
#' @return A list od two object: (i) a data.frame including the connections together
#' with their p-values, and (ii) the DAG with colored edges. If p-values > thr and
#' t-test < 0, the edge is inhibited and it is highlighted in blue; otherwise, if
#' p-values > thr and t-test > 0, the edge is activated and it is highlighted in red. 
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' S. Mohammadi. A new test for the significance of neural network
#' inputs. Neurocomputing 2018; 273: 304-322.
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' 
#' dnn0 <- SEMdnn(ig, data, train=1:nrow(data), cowt = FALSE,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#' 
#' res<- getInputPvalue(dnn0, thr=NULL, verbose=TRUE)
#' table(E(res$dag)$color)
#' }
#' }
#'
#' @export
#' 

getInputPvalue<- function(object, thr = NULL, verbose = FALSE, ...)
{
	model <- object$model
	graph <- object$graph
	A <- as_adjacency_matrix(graph, sparse=FALSE)
	rownames(A) <- paste0("z", rownames(A))
	colnames(A) <- rownames(A)
	est <- NULL
	
	for (j in 1:length(model)) {
	  W <- getPvalue(model[[j]], A)
	  Y <- gsub("z", "", names(W))
	  
	  for (k in 1:length(Y)) {
	   X <- gsub("z", "", rownames(W[[k]]))
	   label<- data.frame(
				lhs = rep(Y[k], length(X)),
				op = "~",
				rhs =  X)
	   est <- rbind(est, cbind(label, W[[k]]))
	  }
	
	}
	rownames(est)<- NULL
	class(est)<- c("lavaan.data.frame","data.frame")
		
	p.adj<- p.adjust(est$pvalue, method="none")
	weight<- sign(est[,4]) * (1 - p.adj)
	df<- data.frame(est[,3],est[,1],weight=weight)
	dag<- graph_from_data_frame(df)
	if (is.null(thr)) thr <- 0.05
	dag<- colorDAG(dag, thr=(1-thr), verbose=verbose)
		
	return(list(est = est, dag = dag))
}

getPvalue <- function(nn.fit, A, ...)
{
	# Test of significance of each yhat fixing other predictors at
	# zero, proposed by S. Mohammadi, Neurocomputing 2018; 304-322
	Z <- scale(nn.fit$data$data)
	vn <- all.vars(nn.fit$old_formula)
	vx <- colnames(nn.fit$data$X)
	vy <- vn[vn %in% vx == FALSE]
	Aj<- as.matrix(A[vx,vy])
	
	Y<- as.matrix(Z[,vy])
	X<- as.matrix(Z[,vx])
	X0<- matrix(0, nrow(X), ncol(X))
	colnames(X0)<- vx

	Yhat<- list()
	for (k in 1:length(vx)) {
	  X0[,k]<- X[,k]
	  Yhatk<- predict(nn.fit, newdata = X0)
	  colnames(Yhatk)<- vy
	  Yhat<- c(Yhat,list(Yhatk))
	} 
	names(Yhat) <- vx

	# Fit a multiple linear model of Yj on Yhatj
	est<- list()
	for (j in 1:length(vy)) {
	  Xj<- data.frame(sapply(Yhat, function(x) x[,j]))
	   if (ncol(Xj) == 1) {
		 nj <- colnames(Xj)
	   }else{
		 nj <- colnames(Xj)[Aj[, j] == 1]
	   }
	  Zj<- data.frame(Y[,j], Xj[,nj])
	  colnames(Zj)<- c(vy[j], nj)
	  f<- paste0(vy[j], "~.")
	  fit<- lm(eval(f), data = Zj)
	  estj<- data.frame(summary(fit)$coefficients)[-1, ]
	  colnames(estj)<- c("est", "se", "t", "pvalue")
	  est<- c(est, list(estj))
	}
	names(est) <- vy

	return(est)
}
