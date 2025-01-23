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
#' @param outcome A character vector (as.factor) of labels for a categorical
#' output (target). If NULL (default), the categorical output (target) will
#' not be considered.
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' Olden's connection weights to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(connection weights)).
#' @param nboot number of bootstrap samples that will be used to compute cheap
#' (lower, upper) CIs for all input variable weights. As a default, nboot = 0.
#' @param hidden hidden units in layers; the number of layers corresponds with
#' the length of the hidden units. As a default, hidden = c(10L, 10L, 10L).
#' @param link A character value describing the activation function to use, which 
#' might be a single length or be a vector with many activation functions assigned
#' to each layer. As a default, link = "selu". 
#' @param bias A logical vector, indicating whether to employ biases in the layers, 
#' which can be either vectors of logicals for each layer (number of hidden layers
#' + 1 (final layer)) or of length one.  As a default, bias = TRUE.
#' @param dropout A numerical value for the dropout rate, which is the probability
#' that a node will be excluded from training.  As a default, dropout = 0. 
#' @param loss A character value specifying the at which the network should 
#' be optimized. For regression problem used in SEMdnn(), the user can specify:
#' (a) "mse" (mean squared error), "mae" (mean absolute error), or "gaussian"
#' (normal likelihood). As a default, loss = "mse".
#' @param validation A numerical value indicating the proportion of the data set 
#' that should be used as a validation set (randomly selected, default = 0). 
#' @param lambda A numerical value indicating the strength of the regularization,
#' \eqn{\lambda}(L1 + L2) for lambda penalty (default = 0). 
#' @param alpha A numerical vector to add L1/L2 regularization into the training. 
#' Set the alpha parameter for each layer to (1-\eqn{\alpha})L1 + \eqn{\alpha}L2. 
#' It must fall between 0 and 1 (default = 0.5). 
#' @param optimizer A character value indicating the optimizer to use for 
#' training the network. The user can specify: "adam" (ADAM algorithm), "adagrad"
#' (adaptive gradient algorithm), "rmsprop" (root mean squared propagation),
#' "rprop” (resilient backpropagation), "sgd" (stochastic gradient descent).
#' As a default, optimizer = "adam".
#' @param lr A numerical value indicating the learning rate given to the optimizer 
#' (default = 0.01). 
#' @param epochs A numerical value indicating the epochs during which the training 
#' is conducted (default = 100). 
#' @param device A character value describing the CPU/GPU device ("cpu", "cuda", "mps")
#' on which the  neural network should be trained on. As a default, device = "cpu".
#' @param ncores number of cpu cores (default = 2)
#' @param early_stopping If set to integer, training will terminate if the loss 
#' increases over a predetermined number of consecutive epochs and apply validation
#' loss when available. Default is FALSE, no early stopping is applied. 
#' @param verbose The training loss values of the DNN model are displayed as output,
#' comparing the training, validation and baseline in the last epoch (default = FALSE). 
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
#' If boot != 0, the function will implement the cheap bootstrapping proposed by
#' Lam (2002) to generate uncertainties, i.e. 90% confidence intervals (90% CIs),
#' for DNN parameters. Bootstrapping can be enabled by setting a small number
#' (1 to 10) of bootstrap samples. Note, however, that the computation can be
#' time-consuming for massive DNNs, even with cheap bootstrapping!
#'
#' @return An S3 object of class "DNN" is returned. It is a list of 5 objects:
#' \enumerate{
#' \item "fit", a list of DNN model objects, including: the estimated covariance 
#' matrix (Sigma), the estimated model errors (Psi), the fitting indices (fitIdx),
#' and the parameterEstimates, i.e., the data.frame of Olden's connection weights. 
#' \item "gest", the data.frame of estimated connection weights (parameterEstimates)
#' of outcome levels, if outcome != NULL.
#' \item "model", a list of all j=1,...,(L-1) fitted MLP network models.
#' \item "graph", the induced DAG of the input graph mapped on data variables.
#' The DAG is colored based on the Olden's connection weights (W), if abs(W) > thr
#' and W < 0, the edge is inhibited and it is highlighted in blue; otherwise, if
#' abs(W) > thr and W > 0, the edge is activated and it is highlighted in red.
#' If the outcome vector is given, nodes with absolute connection weights summed
#' over the outcome levels, i.e. sum(abs(W[outcome levels])) > thr, will be
#' highlighted in pink.
#' \item "data", input data subset mapping graph nodes.
#' }
#'
#' @import SEMgraph
#' @import igraph
#' @importFrom foreach foreach %dopar%
#' @importFrom graphics abline axis boxplot legend lines mtext par points polygon
#' @importFrom grDevices rgb terrain.colors
#' @importFrom stats aggregate as.formula coef coefficients cor density fitted formula 
#'             lm model.matrix predict p.adjust quantile qt reshape rnorm runif sd
#' @importFrom utils tail
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
#' Bioinformatics, 38 (20), 4829–4830. <https://doi.org/10.1093/bioinformatics/btac567>
#' 
#' Lam, H. (2022). Cheap bootstrap for input uncertainty quantification. WSC '22:
#' Proceedings of the Winter Simulation Conference, 2318 - 2329.
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
#' #ncores<- parallel::detectCores(logical = FALSE)
#'
#' start<- Sys.time()
#' dnn0<- SEMdnn(ig, data[train, ], thr = NULL,
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0, epochs = 32, ncores = 2)
#' end<- Sys.time()
#' print(end-start)
#' 
#' #str(dnn0, max.level=2)
#' dnn0$fit$fitIdx
#' parameterEstimates(dnn0$fit)
#' gplot(dnn0$graph)
#' table(E(dnn0$graph)$color)
#'
#' #...with source nodes -> graph layer structure -> sink nodes
#'
#' #Topological layer (TL) ordering
#' K<- c(12,  5,  3,  2,  1,  8)
#' K<- rev(K[-c(1,length(K))]);K
#' 
#' ig1<- mapGraph(ig, type="source"); gplot(ig1)
#'
#' start<- Sys.time()
#' dnn1<- SEMdnn(ig1, data[train, ], thr = NULL,
#' 			hidden = 5*K, link = "selu", bias = TRUE,
#'			validation = 0, epochs = 32, ncores = 2)
#' end<- Sys.time()
#' print(end-start)
#'
#' #Visualization of the neural network structure
#' nn1 <- dnn1$model[[1]][[1]]
#' nplot(nn1, bias=FALSE)
#'
#' #str(dnn1, max.level=2)
#' dnn1$fit$fitIdx
#' mean(dnn1$fit$Psi)
#' parameterEstimates(dnn1$fit)
#' gplot(dnn1$graph)
#' table(E(dnn1$graph)$color)
#' 
#' #...with a categorical outcome, a train set (0.5) and a validation set (0.2)
#' outcome<- factor(ifelse(group == 0, "control", "case")); table(outcome) 
#'
#' start<- Sys.time()
#' dnn2<- SEMdnn(ig, data[train, ], outcome[train], thr = NULL,
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0.2, epochs = 32, ncores = 2)
#' end<- Sys.time()
#' print(end-start)
#' 
#' #str(dnn2, max.level=2)
#' dnn2$fit$fitIdx
#' parameterEstimates(dnn2$fit)
#' gplot(dnn2$graph) 
#' table(E(dnn2$graph)$color)
#' table(V(dnn2$graph)$color)
#' }
#' }
#'
#' @export
#'
SEMdnn <- function(graph, data, outcome = NULL, thr = NULL, nboot = 0,
	 hidden = c(10L, 10L, 10L), link = "relu", bias = TRUE,
	 dropout = 0, loss = "mse", validation = 0, lambda = 0, alpha = 0.5,
	 optimizer = "adam", lr = 0.01, epochs = 100, device = "cpu", ncores = 2,
	 early_stopping = FALSE, verbose = FALSE, ...)
{
	# Set graph and data objects:
	nodes <- colnames(data)[colnames(data) %in% V(graph)$name]
	graph <- induced_subgraph(graph, vids=which(V(graph)$name %in% nodes))
	dag <- graph2dag(graph, data, bap=FALSE) #del cycles & all <->
	data <- data[, V(dag)$name]
	if (!is.null(outcome)) {
	 out <- model.matrix(~outcome-1)
	 colnames(out) <- gsub("outcome", "", colnames(out))
	 data <- cbind(out, data)
	 dag <- mapGraph(dag, type="outcome", C=ncol(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- colnames(out)
	}
	din <- igraph::degree(dag, mode= "in")
	Vx <- V(dag)$name[din == 0]
	Vy <- V(dag)$name[din != 0]
	px <- length(Vx)
	py <- length(Vy)

	# Check whether a compatible GPU is available for computation.
	#use_cuda <- torch::cuda_is_available()
	#device <- ifelse(use_cuda, "cuda", "cpu")
	#fitting a DNN model to predict Y on X
	message("Running SEM model via DNN...")
	res <- list()
	nboot <- nboot + 1
	pb <- pb(nrep = nboot, snow = FALSE)

	  for (b in 1:nboot) { #b=2
	 	#set.seed(runif(1,0,1000))
		set.seed(b)
		if (b == 1) idx<- 1:nrow(data) else idx<- sample(nrow(data),replace = TRUE)
		res[[b]] <- parameterEstimates.DNN(dag, data = data[idx, ], nboot, 
							hidden, link, bias, dropout, loss,
							validation, lambda, alpha, optimizer, lr, epochs,
							device, ncores, early_stopping, verbose)
		pb$tick()
		message(" done.")
	   #print(all.vars(res[[b]]$dnn[[1]]$old_formula));cat("\n")
	  }
	  # str(res[[1]], max.level=2)
	  #str(res, max.level=2)

	# Shat and Sobs matrices :
	Shat <- cor(cbind(data[,Vx], res[[1]]$YHAT[,Vy]))
	rownames(Shat) <- colnames(Shat) <- c(Vx,Vy)
	Sobs <- cor(data[, c(Vx,Vy)])
	E <- Sobs - Shat # diag(E)

	# Fit indices :
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	Psi <- res[[1]]$sigma
	logL <- -0.5 * (sum(log(Psi)) + py * log(nrow(data)))#GOLEM, NeurIPS 2020
	AMSE <- mean(Psi) #average Mean Square Error
	idx <- c(logL = logL, amse = AMSE, rmse = sqrt(AMSE), srmr = SRMR)
	it <- py * epochs
	message("\n", "DNN solver ended normally after ", it, " iterations")
	message("\n", " logL:", round(idx[1],6), "  srmr:", round(idx[4],6))

	# Get connection weights:
	model<- lapply(1:length(res), function(x) res[[x]][[1]])
	gest <- NULL
	#if (cowt) {
	 if (!is.null(outcome)) {
	  gest$lhs <- levels(outcome)
	  dnn <- list(model=model, gest=gest, graph=dag-levels(outcome))
	 }else{
	  dnn <- list(model=model, gest=gest, graph=dag)
	 }
	 cowt <- getConnectionWeight(object=dnn, thr=thr, verbose=FALSE)
	 est <- cowt$est
	 gest<- cowt$gest
	 dag <- cowt$dag
	#} 

	fit <- list(Sigma=Shat, Beta=NULL, Psi=Psi, fitIdx=idx, parameterEstimates=est)
	res <- list(fit=fit, gest=gest, model=model, graph=dag, data=data)
	class(res) <- "DNN"

	return(res)
}

parameterEstimates.DNN <- function(dag, data, nboot, hidden, link, bias, dropout,
							loss, validation, lambda, alpha, optimizer, lr, epochs,
							device, ncores, early_stopping, verbose, ...)
{
	# Set objects:
	Z_train <- scale(data)
	colnames(Z_train) <- paste0("z", colnames(Z_train))
	V(dag)$name <- paste0("z", V(dag)$name)
	L <- buildLevels(dag)
	A <- as_adjacency_matrix(dag, sparse=FALSE) # direct effect matrix

	#require(foreach)
	nrep <- length(L)-1
	cl <- parallel::makeCluster(ncores)
	doSNOW::registerDoSNOW(cl)
	opts <- list(progress = pb(nrep))
	#pb <- txtProgressBar(min=0, max=nrep, style=3)
	#progress <- function(n) setTxtProgressBar(pb, n)
	#opts <- list(progress=progress)

	fit <- foreach(l=1:nrep, .options.snow=opts) %dopar% {
	  vy <- L[[l]]
	  vx <- unlist(L[(l+1):length(L)])
	  Al <- as.matrix( A[vx, vy] )
	  vx <- vx[rowSums(Al) != 0]
	  n <- nrow(Z_train)

	  # Fit a multivariate L1 model of Y on X
	  if (length(vy) > 1 & length(vx) >= n) {
	   if (nboot == 1) {
		x <- as.matrix(Z_train[ ,vx])
		y <- as.matrix(Z_train[ ,vy])
		fit <- glmnet::cv.glmnet(x, y, family="mgaussian")
		C <- coef(fit, s = "lambda.min")
		D <- do.call(cbind, lapply(C, matrix))[-1, ]
		vx <- vx[rowMeans(abs(D)) > 0.01]
	   }else{
	    #set.seed(runif(1,0,1000))
		vx <- vx[sample(1:length(vx), 0.5*n)]
	   }
	  }

	  X <- data.frame(Z_train[,vx]) 
	  if (ncol(X) == 1) colnames(X) <- vx
	  Y<- data.frame(Z_train[,vy])
	  if (ncol(Y) == 1) colnames(Y) <- vy

	  #fitting a dnn model to predict Y on X
	  nn.fit <- cito::dnn(data = Z_train, 
					hidden = hidden,
					activation = link,
					bias = bias, 
					dropout = dropout,
					loss = loss,
					validation = validation,
					lambda = lambda,
					alpha = alpha,
					optimizer = optimizer,
					lr = lr,
					epochs = epochs,
					plot = FALSE,
					verbose = FALSE,
					device = device,
					early_stopping = early_stopping,
					X = X,
					Y = Y)
	}
	#close(pb)
	parallel::stopCluster(cl)
	#str(fit, max.level=1)

	YHAT<- NULL
	sigma<- NULL
	for (l in 1:(length(L)-1)) {

	  if (verbose) {
		epoch<- fit[[l]]$losses[,1]
		train<- fit[[l]]$losses[,2]
		valid<- fit[[l]]$losses[,3]
		base_l<- fit[[l]]$base_loss

		# plot training loss 
		#plot(x=epoch, y=train, type="b", ylim=c(0,base_l+0.05), col="blue",
		#	 xlab = "number of epochs", ylab = "mean square error (mse)",
		#     main=paste0("mse (layer ", l, ") = ", round(train[epochs], 3)))
		#lines(x=epoch, y=valid, type="b", col ="red")
		#abline(h=base_l, col="green")
		#Sys.sleep(1)

		#results of the last iteraction
		if (length(unlist(L[[l]])) > 10) {
		 cat("\nlayer", l,":", unlist(L[[l]])[1:10], "...\n")
		}else{
		 cat("\nlayer", l,":", unlist(L[[l]]), "\n")
		}
		print(cbind(fit[[l]]$losses[epochs,-1],base_l))
	  }
	  
	  #TRAIN predictions and prediction error (MSE)
	  PRED <- predict(fit[[l]], Z_train)
	  pe <- apply((Z_train[ ,L[[l]]] - PRED)^2, 2, mean)
	  sigma <- c(sigma, pe)
	  YHAT <- cbind(YHAT, PRED)
	}
	colnames(YHAT) <- sub(".", "", unlist(L[-length(L)]))
	names(sigma) <- sub(".", "", unlist(L[-length(L)]))

	return(list(dnn = fit, sigma = sigma, YHAT = YHAT))
}

cheapBootCI <- function(est, out, nboot, ...)
{
 	# compute cheap bootsrap CI
	estB <- est[!est[ ,1] %in% out, ]
	gest <- est[est[ ,1] %in% out, ]
	estC <- data.frame()
	
	if (nboot == 1) {
	 estB <- cbind(estB, lower=rep(0, nrow(estB)), upper=rep(0, nrow(estB)))
	 
	 if (length(out) > 0) {
	  
	  C <- aggregate(abs(gest[ ,4])~gest[ ,3],FUN="sum")
	  estC <- data.frame(
				lhs = rep("outcome", nrow(C)),
				op = "~",
				rhs = C[ ,1],
				est = C[ ,2],
				lower = rep(0, nrow(C)),
				upper = rep(0, nrow(C)))
	  colnames(estC)[4] <- colnames(est)[4]
	  class(gest) <- c("lavaan.data.frame","data.frame")

	 }else{
	  gest <- NULL
	 }
	}

	if (nboot > 1) {
	  t <- qt(p=0.05, df=nboot-1, lower.tail=FALSE); t #90%CI
	  
	  for (j in 1:nrow(estB)) { # j=1
	   bj <- estB[j,4]
	   dj <- (estB[j,5:(nboot+3)] - bj)^2
	   estB$lower[j] <- bj - t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	   estB$upper[j] <- bj + t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	  }
	  estB <- estB[ ,c(1:4,nboot+4,nboot+5)]
	 	  
	  if (length(out) > 0) {
	   
	   L <- list()
	   for (l in 1:length(out)) L[[l]]<- abs(gest[gest$lhs == out[l],4:ncol(gest)])
	   C <- Reduce('+', L)
	   for (j in 1:nrow(C)) {
	    cj <- C[j,1]
		dj <- (C[j,2:ncol(C)] - cj)^2
	    C$lower[j] <- cj - t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	    C$upper[j] <- cj + t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	   }
	   estC <- data.frame(
				lhs = rep("outcome", nrow(C)),
				op = "~",
				rhs = gest$rhs[1:nrow(C)],
				C[ ,c(1,ncol(C)-1,ncol(C))])
		
	   for (j in 1:nrow(gest)) { # j=1
	    gj <- gest[j,4]
		dj <- (gest[j,5:(nboot+3)] - gj)^2
	    gest$lower[j] <- gj - t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	    gest$upper[j] <- gj + t*sqrt(sum(dj, na.rm=TRUE)/(nboot-1))
	   }   
	   gest <- gest[ ,c(1:4,nboot+4,nboot+5)]
	   class(gest) <- c("lavaan.data.frame","data.frame")

	  }else{
	   gest <- NULL
	  }
	}
	
	class(estB) <- c("lavaan.data.frame","data.frame")
	class(estC) <- c("lavaan.data.frame","data.frame")

	return(list(estB = estB, estC = estC, gest = gest))
}

colorDAG <- function(dag, est, out, nboot, thr=NULL, verbose=FALSE, ...)
{
	estB <- est[[1]]
	estC <- est[[2]]

	# set node colors
	if (length(out) > 0) {
	 dag0 <- dag - out
	 V(dag0)$weight <- estC[ ,4][match(V(dag0)$name, estC$rhs)]
	 V(dag0)$lower <- estC[ ,5][match(V(dag0)$name, estC$rhs)]
	 V(dag0)$color <- ifelse(V(dag0)$lower > 0, "pink", "white")
	 if (nboot == 1) {
	  maxV <- max(abs(V(dag0)$weight), na.rm=TRUE)
	  thrV <- ifelse(is.null(thr), 0.5*maxV, thr*maxV)
	  V(dag0)$color <- ifelse(V(dag0)$weight > thrV, "pink", "white")
	 }
	}else{
	 dag0 <- dag
	}

	# set edge colors
	E1 <- paste0(estB$rhs,"|",estB$lhs)
	E0 <- attr(E(dag0), "vnames")
	E(dag0)$weight <- estB[ ,4][match(E0, E1)]
	E(dag0)$lower <- estB[ ,5][match(E0, E1)]
	E(dag0)$upper <- estB[ ,6][match(E0, E1)]
	enames <- attr(E(dag0), "vnames")
	Er <- enames[E(dag0)$weight < 0 & E(dag0)$upper < 0]
	Ea <- enames[E(dag0)$weight > 0 & E(dag0)$lower > 0]
	if (nboot == 1) {
	 maxE <- max(abs(E(dag0)$weight), na.rm=TRUE)
	 thrE <- ifelse(is.null(thr), 0.5*maxE, thr*maxE)
	 Er <- enames[E(dag0)$weight < 0 & abs(E(dag0)$weight) > thrE]
	 Ea <- enames[E(dag0)$weight > 0 & abs(E(dag0)$weight) > thrE]
	}
	E(dag0)$color <- ifelse(attr(E(dag0), "vnames") %in% 
			Er, "royalblue3", ifelse(attr(E(dag0), "vnames") %in%
			Ea, "red2", "gray50"))
	E(dag0)$width <- ifelse(E(dag0)$color == "gray50", 1, 2)

	if (verbose) gplot(dag0)
		
	return(list(est=rbind(estB[match(E0, E1), ], estC), gest=est[[3]], dag=dag0))
}

#' @title SEM-based out-of-sample prediction using layer-wise DNN
#'
#' @description Predict method for DNN objects.
#'
#' @param object A model fitting object from \code{SEMdnn()} function. 
#' @param newdata A matrix containing new data with rows corresponding to
#' subjects, and columns to variables.
#' @param newoutcome A new character vector (as.factor) of labels for a categorical
#' output (target) (default = NULL).
#' @param verbose Print predicted out-of-sample MSE values (default = FALSE).
#' @param ... Currently ignored.
#'
#' @return A list of three objects:
#' \enumerate{
#' \item "PE", vector of the amse = average MSE over all (sink and mediators)
#' graph nodes; r2 = 1 - amse; and srmr= Standardized Root Means Square Residual
#' between the out-of-bag correlation matrix and the model correlation matrix.
#' \item "mse", vector of the Mean Squared Error (MSE) for each out-of-bag
#' prediction of the sink and mediators graph nodes.
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
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' group<- alsData$group 
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' #ncores<- parallel::detectCores(logical = FALSE)
#'
#' start<- Sys.time()
#' dnn0 <- SEMdnn(ig, data[train, ],
#' 			# hidden = 5*K, link = "selu", bias = TRUE, 
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE, 
#' 			validation = 0, epochs = 32, ncores = 2)
#' end<- Sys.time()
#' print(end-start)
#' pred.dnn <- predict(dnn0, data[-train, ], verbose=TRUE)
#' 
#' # SEMrun vs. SEMdnn MSE comparison
#' sem0 <- SEMrun(ig, data[train, ], algo="ricf", n_rep=0)
#' pred.sem <- predict(sem0, data[-train,], verbose=TRUE)
#' 
#' #...with a categorical (as.factor) outcome
#' outcome <- factor(ifelse(group == 0, "control", "case")); table(outcome) 
#' 
#' start<- Sys.time()
#' dnn1 <- SEMdnn(ig, data[train, ], outcome[train],
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0,  epochs = 32, ncores = 2)
#' end<- Sys.time()
#' print(end-start)
#'
#' pred <- predict(dnn1, data[-train, ], outcome[-train], verbose=TRUE)
#' yhat <- pred$Yhat[ ,levels(outcome)]; head(yhat)
#' yobs <- outcome[-train]; head(yobs)
#' classificationReport(yobs, yhat, verbose=TRUE)$stats
#' }
#' }
#'
#' @method predict DNN
#' @export
#' @export predict.DNN
#' 
predict.DNN <- function(object, newdata, newoutcome = NULL, verbose=FALSE, ...)
{
	stopifnot(inherits(object$model[[1]][[1]], "citodnn"))
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	if (!is.null(newoutcome)) {
	 out <- model.matrix(~newoutcome-1)
	 colnames(out) <- gsub("newoutcome", "", colnames(out))
	 newdata <- cbind(out, newdata)
	}
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))

	dnn.fit <- object$model[[1]]
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

	yobs <- Z_test[, yn]
	PE <- colMeans((yobs - yhat)^2)
	PE <- ifelse(PE > 1, NA, PE)
	pe <- mean(PE, na.rm = TRUE)
	colnames(yhat) <- sub(".", "", yn)
	names(PE) <- colnames(yhat)
		
	Shat<- object$fit$Sigma
	V <- colnames(Shat)
	Sobs <- cor(newdata[, V])
	E <- Sobs - Shat
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	
	if (verbose) print(c(amse=pe, r2=1-pe, srmr=SRMR))
	
	return(list(PE=c(amse=pe, r2=1-pe, srmr=SRMR), mse=PE, Yhat=yhat))
}

#' @title Connection Weight method for neural network variable importance
#'
#' @description The function computes the matrix multiplications of hidden
#' weight matrices (Wx,...,Wy), i.e., the product of the raw input-hidden and
#' hidden-output connection weights between each input and output neuron and
#' sums the products across all hidden neurons, as proposed by Olden (2004).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' Olden's connection weights to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(connection weights)).
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
#' @return A list of three object: (i) est: a data.frame including the connections together
#' with their connection weights(W), (ii) gest: if the outcome vector is given, a data.frame
#' of connection weights for outcome lavels, and (iii) dag: DAG with colored edges/nodes. If
#' abs(W) > thr and W < 0, the edge W > 0, the edge is activated and it is highlighted
#' in red. If the outcome vector is given, nodes with absolute connection weights summed
#' over the outcome levels, i.e. sum(abs(W[outcome levels])) > thr, will be highlighted
#' in pink.
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
#' #ncores<- parallel::detectCores(logical = FALSE)
#' dnn0<- SEMdnn(ig, data, outcome = NULL, thr = NULL,
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0,  epochs = 32, ncores = 2)
#' 
#' cw05<- getConnectionWeight(dnn0, thr = 0.5, verbose = TRUE)
#' table(E(cw05$dag)$color)
#' }
#' }
#'
#' @export
#' 
getConnectionWeight <- function(object, thr = NULL, verbose = FALSE, ...)
{
	# Matrix product of input-hidden, hidden-output weights,
	# proposed by Olden et al. Ecol. Model. 2004;389–397
	stopifnot(inherits(object$model[[1]][[1]], "citodnn"))
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}

	nboot<- length(object$model)
	est_boot <- list()
	#pb <- pb(nrep = nboot , snow = FALSE)
	
	for(b in 1:nboot) {
	   #pb$tick()
	   model <- object$model[[b]]
	   M <- length(model)
	   nrep <- ifelse(inherits(object, "ML"), M-1, M)
	   est <- NULL
	   for (j in 1:nrep) {
		 vn <- all.vars(model[[j]]$old_formula)
		 vx <- colnames(model[[j]]$data$X)
		 vy <- vn[vn %in% vx == FALSE]
		 W <- coef(model[[j]])[[1]]
		 w <- t(W[[1]])
		 for (k in seq(3, length(W), by=2)) { 
			w <- w %*% t(W[[k]])
			if (length(W) == 4) break
		 }
		 rownames(w) <- vx <- sub(".", "", vx)
		 colnames(w) <- vy <- sub(".", "", vy)
		 for (k in 1:length(vy)) {
			label <- data.frame(
					  lhs = vy[k],
					  op = "~",
					  rhs = vx)
			est <- rbind(est, cbind(label, w[vx,vy[k]]))
	     }
	   }
	   rownames(est) <- NULL
	   colnames(est)[4] <- "weight"
	   
	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

#' @title Gradient Weight method for neural network variable importance
#'
#' @description  The function computes the gradient matrix, i.e., the average
#' conditional effects of the input variables w.r.t the neural network model,
#' as discussed by Amesöder et al (2024).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' gradient weights to color the graph. If thr = NULL (default), the threshold
#' is set to thr = 0.5*max(abs(gradient weights)).
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
#' @return A list of three object: (i) est: a data.frame including the connections together
#' with their gradient weights, (ii) gest: if the outcome vector is given, a data.frame of
#' gradient weights for outcome lavels, and (iii) dag: DAG with colored edges/nodes. If
#' abs(grad) > thr and grad < 0, the edge is inhibited and it is highlighted in blue;
#' otherwise, if abs(grad) > thr and grad > 0, the edge is activated and it is highlighted
#' in red. If the outcome vector is given, nodes with absolute connection weights summed
#' over the outcome levels, i.e. sum(abs(grad[outcome levels])) > thr, will be highlighted
#' in pink.
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
#' #ncores<- parallel::detectCores(logical = FALSE)
#' dnn0<- SEMdnn(ig, data, outcome = NULL, thr = NULL,
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0,  epochs = 32, ncores = 2)
#' 
#' gw05<- getGradientWeight(dnn0, thr = 0.5, verbose = TRUE)
#' table(E(gw05$dag)$color)
#' }
#' }
#'
#' @export
#' 
getGradientWeight<- function(object, thr = NULL, verbose = FALSE, ...)
{
	stopifnot(inherits(object$model[[1]][[1]], "citodnn"))
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}

	nboot<- length(object$model)
	est_boot <- list()
	pb <- pb(nrep = nboot , snow = FALSE)
	
	for(b in 1:nboot) {
	  pb$tick()
	  model<- object$model[[b]]
	  M <- length(model)
	  nrep <- ifelse(inherits(object, "ML"), M-1, M)
	  est <- NULL

	  for (j in 1:nrep) {
		#pb$tick()
		vn <- all.vars(model[[j]]$old_formula)
		vx <- colnames(model[[j]]$data$X)
		vy <- vn[vn %in% vx == FALSE]
		if (length(vx) > 1) {
			W <- quiet(print(cito::conditionalEffects(model[[j]])))
		}else{
			W <- matrix(0.5, nrow=1, ncol=length(vy))
		}
		rownames(W) <- vx <- sub(".", "", vx)
		colnames(W) <- vy <- sub(".", "", vy)
		for (k in 1:length(vy)) {
			label <- data.frame(
					lhs = vy[k],
					op = "~",
					rhs = vx)
			est <- rbind(est, cbind(label, W[vx,vy[k]]))
		}
	  }
	  rownames(est) <- NULL
	  colnames(est)[4] <- "grad"
	
	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
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
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' t-test values to color the graph. If thr = NULL (default), the threshold
#' is set to thr = 0.5*max(abs(t-test values)).
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details A neural network with an arbitrary architecture is trained, 
#' taking into account factors like the number of neurons, hidden layers, 
#' and activation function. Then, network's output is simulated to get 
#' the predicted values of the output variable, fixing all the inputs
#' (with the exception of one nonconstant input variable) at their mean
#' values. Subsequently, the network's predictions are stored after this
#' process is completed for each input variable. As last step, multiple
#' regression analysis is applied node-wise (mapping the input DAG) on the
#' observed output nodes with the predicted values of the input nodes as
#' explanatory variables. The statistical significance of the coefficients
#' is evaluated with the standard t-student critical values, which represent
#' the importance of the input variables. 
#' 
#' @return A list of three object: (i) est: a data.frame including the connections together
#' with their t_test weights, (ii) gest: if the outcome vector is given, a data.frame of
#' t_test weights for outcome lavels, and (iii) dag: DAG with colored edges/nodes. If
#' abs(t_test) > thr and t_test < 0, the edge is inhibited and it is highlighted in blue;
#' otherwise, if abs(t_test) > thr and t_test > 0, the edge is activated and it is highlighted
#' in red. If the outcome vector is given, nodes with absolute connection weights summed
#' over the outcome levels, i.e. sum(abs(t_test[outcome levels])) > thr, will be highlighted
#' in pink.
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
#' #ncores<- parallel::detectCores(logical = FALSE)
#' dnn0 <- SEMdnn(ig, data, outcome = NULL, thr = NULL,
#' 			#hidden = 5*K, link = "selu", bias = TRUE,
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			validation = 0,  epochs = 32, ncores = 2)
#' 
#' st05<- getSignificanceTest(dnn0, thr = 2, verbose = TRUE)
#' table(E(st05$dag)$color)
#' }
#' }
#'
#' @export
#' 
getSignificanceTest<- function(object, thr = NULL, verbose = FALSE, ...)
{
	# Test of significance of each yhat fixing other predictors at
	# zero, proposed by S. Mohammadi, Neurocomputing 2018; 304-322
	stopifnot(inherits(object$model[[1]][[1]], "citodnn"))
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}

	nboot<- length(object$model)
	est_boot <- list()
	pb <- pb(nrep = nboot , snow = FALSE)
	
	for(b in 1:nboot) {
	  pb$tick()
	  model<- object$model[[b]]
	  M <- length(model)
	  nrep <- ifelse(inherits(object, "ML"), M-1, M)
	  est <- NULL
	
	  for (j in 1:nrep) {
		W <- getTestValue(model[[j]])
		if (is.null(W)) {
		 message("\nWARNING: Model (", j,") skipped. Currently less samples than features not supported!")
		 next
		}
		Y <- sub(".", "", names(W))
		for (k in 1:length(Y)) {
		 X <- sub(".", "", rownames(W[[k]]))
		 label<- data.frame(
					lhs = rep(Y[k], length(X)),
					op = "~",
					rhs =  X)
		 est <- rbind(est, cbind(label, W[[k]]))
		}
	  }
	  rownames(est) <- NULL
	  est<- est[ ,c(1:3,6)]
	  colnames(est)[4] <- "t_test"
		
	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

getTestValue <- function(nn.fit, ...)
{
	Z<- scale(nn.fit$data$data)
	vn<- all.vars(nn.fit$old_formula)
	vx<- colnames(nn.fit$data$X)
	vy<- vn[vn %in% vx == FALSE]
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
	 Xj <- as.matrix(sapply(Yhat, function(x) x[,j]))
	 yj <- as.numeric(Y[,j])
	 if (ncol(Xj) >= nrow(Xj)) return(est = NULL)
	 Zj <- data.frame(Y[,j], Xj)
	 colnames(Zj) <- c(vy[j], vx)
	 f <- paste0(vy[j], "~.")
	 fit <- lm(eval(f), data = Zj)
	 estj <- data.frame(summary(fit)$coefficients)[-1, ]
	 colnames(estj) <- c("est", "se", "t", "pvalue")
	 est <- c(est, list(estj))
	}
	names(est) <- vy

	return(est)
}
