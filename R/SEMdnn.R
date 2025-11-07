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

#' @title SEM train with Deep Neural Netwok (DNN) models
#'
#' @description The function builds four Deep Neural Networks (DNN) models based
#' on the topological structure of the input graph using the 'torch' language.
#' The \pkg{torch} package is native to R, so it's computationally efficient
#' and the installation is very simple, as there is no need to install Python
#' or any other API, and DNNs can be trained on CPU, GPU and MacOS GPUs.
#'
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
#' @param algo A character value, indicating the DNN algorithm: "nodewise",
#' "layerwise" (default), "structured", or "neuralgraph" (see details).
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
#' (a) "mse" (mean squared error), "mae" (mean absolute error), or "nnl"
#' (negative log-likelihood). As a default, loss = "mse".
#' @param validation A numerical value indicating the proportion of the data set 
#' that should be used as a validation set (randomly selected, default = 0). 
#' @param lambda A numerical value, indicating the strength of the regularization,
#' \eqn{\lambda}(L1 + L2) for lambda penalty (default = 0). 
#' @param alpha A numerical value, add L1/L2 regularization into the training. 
#' Set the alpha parameter for each layer to (1-\eqn{\alpha})L1 + \eqn{\alpha}L2. 
#' It must fall between 0 and 1 (default = 0.5). 
#' @param optimizer A character value, indicating the optimizer to use for 
#' training the network. The user can specify: "adam" (ADAM algorithm), "adagrad"
#' (adaptive gradient algorithm), "rmsprop" (root mean squared propagation),
#' "rprop” (resilient backpropagation), "sgd" (stochastic gradient descent).
#' As a default, optimizer = "adam".
#' @param lr A numerical value, indicating the learning rate given to the optimizer 
#' (default = 0.01).
#' @param batchsize Number of samples that are used to calculate one learning rate
#' step (default = 1/10 of the training data). 
#' @param burnin Training is aborted if the trainings loss is not below the baseline
#' loss after burnin epochs (default = 30).
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' Olden's connection weights to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(connection weights)).
#' @param nboot number of bootstrap samples that will be used to compute cheap
#' (lower, upper) CIs for all input variable weights. As a default, nboot = 0.
#' @param epochs A numerical value indicating the epochs during which the training 
#' is conducted (default = 100). 
#' @param patience A numeric value, training will terminate if the loss 
#' increases over a predetermined number of consecutive epochs and apply validation
#' loss when available. Default patience = 100, no early stopping is applied. 
#' @param device A character value describing the CPU/GPU device ("cpu", "cuda", "mps")
#' on which the  neural network should be trained on. As a default, device = "cpu".
#' @param verbose The training loss values of the DNN model are displayed as output,
#' comparing the training, validation and baseline in the last epoch (default = FALSE). 
#' @param ... Currently ignored.
#'
#' @details Four Deep Neural Networks (DNNs) are trained with \code{SEMdnn()}.
#'
#' If algo = "nodewise", a set of DNN models is performed equation-by-equation
#' (r=1,...,R) times, where R is the number of response (outcome) variables (i.e.,
#' nodes in the input graph with non-zero incoming connectivity) and predictor (input)
#' variables are nodes with a direct edge to the outcome nodes, as poposed by various
#' authors in causal discovery methods (see Zheng et al, 2020). Note, that model
#' learning can be time-consuming for large graphs and large R outcomes.
#'
#' If algo = "layerwise" (default), a set of DNN models is defined based on the topological
#' layer structure (j=1,…,L) from sink to source nodes of the input graph. In each
#' iteration, the response (output) variables, y are the nodes in the j=1,...,(L-1)
#' layer, and the predictor (input) variables, x are the nodes belonging to successive:
#' (j+1),...,L layers, which are linked with a direct edge to the response variables
#' (see Grassi & Tarantino, 2025).
#'
#' If algo = "structured", a Structured Neural Network (StrNN) is defined with input
#' and output units equal to D, the number of the nodes. The algorithm uses the
#' prior knowledge of the input graph to build the neural network architecture via a
#' per-layer masking of the neural weights (i.e., W1 * M1, W2 * M2, ..., WL *ML), with
#' the constraint that (W1 * M1) x (W2 * M2) x ... x (WL * ML) = A, where A is the
#' adjacency matrix of the input graph (see Chen et al, 2023).
#'
#' If algo = "neuralgraph", a Neural Graphical Model (NGM) is generated. As StrNN input
#' and output units are equal to D, the number of the nodes. The prior knowledge of the
#' input graph is used to compute the product of the absolute value of the neural weights
#' (i.e., W = |W1| x |W2| x ... x |WL|), under the constraint that log(W * Ac) = 0,
#' where Ac represents the complement of the adjacency matrix A of input graph, which
#' essentially replaces 0 by 1 and vice-versa (see Shrivastava & Chajewska, 2023).
#'
#' Each DNN model (R for "nodewise", L<R for "layerwise", and 1 for "structured" and
#' "neuralgraph") is a Multilayer Perceptron (MLP) network, where every neuron node
#' is connected to every other neuron node in the hidden layer above and every other
#' hidden layer below. Each neuron's value is determined by calculating a weighted
#' summation of its outputs from the hidden layer before it, and then applying an
#' activation function.  The calculated value of every neuron is used as the input
#' for the neurons in the layer below it, until the output layer is reached.
#'
#' If boot != 0, the function will implement the cheap bootstrapping proposed by
#' Lam (2002) to generate uncertainties (i.e., bootstrap \code{90\%CIs}) for DNN
#' parameters. Bootstrapping can be enabled by setting a small number (1 to 10) of
#' bootstrap samples. Note, however, that the computation can be time-consuming for
#' massive DNNs, even with cheap bootstrapping!
#'
#' @return An S3 object of class "DNN" is returned. It is a list of 5 objects:
#' \enumerate{
#' \item "fit", a list of DNN model objects, including: the estimated covariance 
#' matrix (Sigma), the estimated model errors (Psi), the fitting indices (fitIdx),
#' and the parameterEstimates, i.e., the data.frame of Olden's connection weights. 
#' \item "gest", the data.frame of estimated connection weights (parameterEstimates)
#' of outcome levels, if outcome != NULL.
#' \item "model", a list of all MLP network models fitted by torch.
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
#' @importFrom graphics abline axis boxplot legend lines mtext par points polygon
#' @importFrom grDevices rgb terrain.colors
#' @importFrom stats aggregate as.formula coef coefficients cor density fitted formula 
#'             lm model.matrix na.omit predict p.adjust quantile qt reshape rnorm runif sd
#' @importFrom utils tail
#' @importFrom torch torch_is_installed    
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' Zheng, X., Dan, C., Aragam, B., Ravikumar, P., Xing E. (2020). Learning sparse
#' nonparametric dags. International conference on artificial intelligence and statistics,
#' PMLR, 3414-3425. https://doi.org/10.48550/arXiv.1909.13189
#'
#' Grassi, M., Tarantino, B. (2025). SEMdag: Fast learning of Directed Acyclic Graphs via
#' node or layer ordering. PLoS ONE 20(1): e0317283. https://doi.org/10.1371/journal.pone.0317283
#'
#' Chen A., Shi, R.I., Gao, X., Baptista, R., Krishnan, R.G. (2023). Structured neural
#' networks for density estimation and causal inference. Advances in Neural Information
#' Processing Systems, 36, 66438-66450. https://doi.org/10.48550/arXiv.2311.02221
#'
#' Shrivastava, H., Chajewska, U. (2023). Neural graphical models. In European Conference
#' on Symbolic and Quantitative Approaches with Uncertainty (pp. 284-307). Cham: Springer
#' Nature Switzerland. https://doi.org/10.48550/arXiv.2210.00453
#'
#' Lam, H. (2022). Cheap Bootstrap for Input Uncertainty Quantification. Winter Simulation 
#' Conference (WSC), Singapore, 2022, pp. 2318-2329. https://doi.org/10.1109/WSC57314.2022.10015362
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
#' dnn0<- SEMdnn(ig, data[train, ], algo = "layerwise",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
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
#' dnn1<- SEMdnn(ig1, data[train, ], algo = "layerwise",
#' 			hidden = 5*K, link = "selu", bias = TRUE,
#'			epochs = 32, patience = 10, verbose = TRUE)
#' end<- Sys.time()
#' print(end-start)
#'
#' #Visualization of the neural network structure
#' nplot(dnn1, hidden = 5*K, bias = FALSE)
#'
#' #str(dnn1, max.level=2)
#' dnn1$fit$fitIdx
#' mean(dnn1$fit$Psi)
#' parameterEstimates(dnn1$fit)
#' gplot(dnn1$graph)
#' table(E(dnn1$graph)$color)
#' 
#' #...with a categorical outcome
#' outcome<- factor(ifelse(group == 0, "control", "case")); table(outcome) 
#'
#' start<- Sys.time()
#' dnn2<- SEMdnn(ig, data[train, ], outcome[train], algo = "layerwise",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
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
SEMdnn <- function(graph, data, outcome = NULL, algo = "layerwise",
	hidden = c(10L, 10L, 10L), link = "selu", bias = TRUE, dropout = 0,
	loss = "mse", validation = 0, lambda = 0, alpha = 0.5, 
	optimizer = "adam", lr = 0.01, batchsize = NULL, burnin = 30, thr = NULL,
	nboot = 0, epochs = 100, patience = 100, device = "cpu", verbose = FALSE, ...)
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
	message("Running SEM model via DNN...")
	
	if (nboot == 0) {
	  res <- list()
	  nboot <- 1 + nboot
	  pb0 <- pb(nrep = nboot, snow = FALSE)

	   for (b in 1:nboot) { #b=1
	 	 #set.seed(runif(1,0,1000))
		 set.seed(b)
		 if (b == 1) idx<- 1:nrow(data) else idx<- sample(nrow(data),replace = TRUE)

		 res[[b]] <- parameterEstimates.DNN(dag, data[idx, ], algo,
						hidden, link, bias, dropout,
						loss, validation, lambda, alpha,
						optimizer, lr, batchsize, burnin,
						epochs, patience, device, verbose)
		
		 pb0$tick()
		 message(" done.")
	   }
	   # str(res, max.level=2)

	  Yhat <- res[[1]]$YHAT[ ,Vy]
	  Psi <- res[[1]]$sigma[Vy]
	  Shat <- cor(cbind(data[ ,Vx], Yhat))
	  rownames(Shat) <- colnames(Shat) <- c(Vx,Vy)
	  model <- list(res[[1]]$dnn)
	}
	
	if (nboot > 1) {
	  message("\nRunning loop: 1 raw sample + ", nboot, " bootstap samples")
	  nboot <- 1 + nboot
	  ncores <- parallel::detectCores(logical = FALSE)
	  backend <- parabar::start_backend(ncores)
	  parabar::export(backend, ls(environment()), environment())
	  #parabar::configure_bar(type = "modern", format = "[:bar] :percent")

	  res <- parabar::par_lapply(backend, x=1:nboot, function(x) { #x=1
        #set.seed(runif(1,0,1000))
	 	set.seed(x)
	 	if (x == 1) idx<- 1:nrow(data) else idx<- sample(nrow(data),replace = TRUE)
 		
		m = do.call(SEMdeep::SEMdnn, args = list(
          graph = graph, data = data[idx, ], outcome = outcome, algo = algo,
		  hidden = hidden, link = link, bias = bias, dropout = dropout,
		  loss = loss, validation = validation, lambda = lambda, alpha = alpha,
		  optimizer = optimizer, lr = lr, batchsize = batchsize, shuffle = TRUE,
		  baseloss = NULL, burnin = burnin, thr = thr, nboot = 0,
	  	  epochs = epochs, patience = patience, device = device, verbose = FALSE
		))

		return(m)
	  })

 	  parabar::stop_backend(backend) # str(res[[1]], max.level=1)
	  
	  Psi <- res[[1]]$fit$Psi
	  Shat <- res[[1]]$fit$Sigma
	  model <- sapply(1:length(res), function(x) res[[x]]$model)
	  #b=1; l=1; model[[b]][[l]]$net
	  
	  #Reconstructing Torch Models
	  for (b in 1:nboot) { #b=1
	    for (l in 1:length(model[[1]])) {	#l=1
		  # Set parameters
		  vx <- ncol(model[[b]][[l]]$data$X)
		  vy <- ncol(model[[b]][[l]]$data$Y)
		  p <- model[[b]][[l]]$param
		  A <- model[[b]][[l]]$data$A
		  
		  # Create a new model instance
		  net <- build_model(algo=p[[1]], input=vx, output=vy, hidden=p[[2]],
						activation=p[[3]], bias=p[[4]], dropout=p[[5]], A=A)

		  # Convert saved R arrays back to torch tensors
		  state_dict <- model[[b]][[l]]$state_dict[[1]]
		  net$load_state_dict(lapply(state_dict, torch::torch_tensor))
		  
		  # Load weights into the model
		  net$load_state_dict(state_dict)
		  model[[b]][[l]]$net <- net
	    }
	  }
	  message("\n done.")
	}
	
	# Fit indices :
	Sobs <- cor(data[ ,c(Vx,Vy)])
	E <- Sobs - Shat # diag(E)
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2 ))
	logL <- -0.5 * (sum(log(Psi)) + py * log(nrow(data)))#GOLEM, NeurIPS 2020
	AMSE <- mean(Psi) #average Mean Square Error
	idx <- c(logL = logL, amse = AMSE, rmse = sqrt(AMSE), srmr = SRMR)
	it <- length(model[[1]]) * epochs * nboot
	message("\n", "DNN solver ended normally after ", it, " iterations")
	message("\n", " logL:", round(idx[1],6), "  srmr:", round(idx[4],6))
	
	# Get connection weights:
	gest <- NULL
	if (!is.null(outcome)) {
	  gest$lhs <- levels(outcome)
	  dnn <- list(model=model, gest=gest, graph=dag-levels(outcome))
	}else{
	  dnn <- list(model=model, gest=gest, graph=dag)
	}
	cowt <- getConnectionWeight(object=dnn, thr=thr, verbose=FALSE)
	est <- cowt$est   # head(est)
	gest <- cowt$gest # head(gest)
	dag <- cowt$dag   # gplot(dag)

	# Output an S3 list
	fit <- list(Sigma=Shat, Beta=NULL, Psi=Psi, fitIdx=idx, parameterEstimates=est)
	res <- list(fit=fit, gest=gest, model=model, graph=dag, data=data)

	class(res) <- "DNN"
			
	return(res)
}

parameterEstimates.DNN <- function(dag, data, algo,
								   hidden, link, bias, dropout,
								   loss, validation, lambda, alpha,
								   optimizer, lr, batchsize, burnin,
								   epochs, patience, device, verbose)
{
	# Set objects:
	Z_train <- scale(data)
	colnames(Z_train) <- paste0("z", colnames(Z_train))
	V(dag)$name <- paste0("z", V(dag)$name)
	A <- as_adjacency_matrix(dag, sparse=FALSE) # raw:x; col:y
	
	if (algo == "nodewise" | algo == "layerwise") {
	  L <- buildLevels(dag)
	  pe<- igraph::as_data_frame(dag)[ ,c(1,2)]
	  y <- split(pe, f=pe$to)
	  
	  nrep <- ifelse(algo == "nodewise", length(y), length(L)-1)
	  #cl <- parallel::makeCluster(ncores)
	  #opb<- pbapply::pboptions(type = "timer", style = 2)
	  pb0 <- pb(nrep = nrep, snow = FALSE)
	  
	  #fit <- pbapply::pblapply(1:nrep, function(x){
	  fit <- lapply(1:nrep, function(x){ #x=1
		pb0$tick()

		if (algo == "nodewise") {
		 vy <- names(y)[x]
		 vx <- y[[x]][,1]
		}
		if (algo == "layerwise") {
		 vy <- L[[x]]
		 vx <- unlist(L[(x+1):length(L)])
		 vx <- vx[rowSums(as.matrix(A[vx,vy])) != 0]
		}
		n <- nrow(Z_train)
		if (length(vx) >= n) {
		 set.seed(1)
		 vx <- vx[sample(1:length(vx), round(0.9*n))]
		}

		X <- data.frame(Z_train[,vx]) #head(X)
		if (ncol(X) == 1) colnames(X) <- vx
		Y<- data.frame(Z_train[,vy]) #head(Y)
		if (ncol(Y) == 1) colnames(Y) <- vy
		
		nn.fit <- train_model(data = Z_train,
							algo = algo,
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
							batchsize = batchsize,
							shuffle = TRUE,
							baseloss = NULL,
							burnin = burnin,
			
							epochs = epochs,
							patience = patience,
							device = device,
							verbose = verbose,
			
							X = X,
							Y = Y,
							A = NULL)
		})
		names(fit) <- paste0("L", 1:length(fit))
		#}, cl=NULL)
		#str(fit, max.level=1)

	  YHAT <- NULL
	  sigma <- NULL
	  
	  for (l in 1:nrep) {
	   #print results of the last interaction
	   if (verbose) {
		if (algo == "layerwise") {
		 if (length(unlist(L[[l]])) > 10) {
		  cat("\nlayer", l,":", unlist(L[[l]])[1:10], "...\n")
		 } else {
		  cat("\nlayer", l,":", unlist(L[[l]]), "\n")
		 }
		 print(fit[[l]]$loss)
		}
		if (algo == "nodewise") {
		 cat("\nnode", l,":", names(y)[l], "\n")
		 print(fit[[l]]$loss)
		}
	   }
	   
	   #TRAIN predictions and prediction error (MSE)
	   if (algo == "nodewise") L[[l]] <- names(y)[l]
	   PRED <- extract_fitted_values(fit[[l]], Z_train)$y_hat #head(PRED)
	   pe <- apply((Z_train[ ,L[[l]]] - PRED)^2, 2, mean)
	   sigma <- c(sigma, pe)
	   YHAT <- cbind(YHAT, PRED) 
	  }
	}
	
	if (algo == "structured" | algo == "neuralgraph") {
	  A <- A[colnames(Z_train), colnames(Z_train)]
	  vx <- which(colSums(A) == 0) # vx 
	  #vy <- which(rowSums(A) == 0) # vy
	  if (algo == "structured") bias <- TRUE

		nn.fit <- train_model(data = Z_train,
							algo = algo,
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
							batchsize = batchsize,
							shuffle = TRUE,
							baseloss = NULL,
							burnin = burnin,

							epochs = epochs,
							patience = patience,
							device = device,
							verbose = verbose,

							X = Z_train,
							Y = Z_train,
							A = A)

		fit <- list(L1 = nn.fit)
		#str(fit, max.level=1); class(fit$model)
	  
	  #TRAIN predictions and prediction error (MSE)
	  PRED <- extract_fitted_values(nn.fit, Z_train)$y_hat #head(PRED)
	  sigma <- apply(as.matrix(Z_train - PRED)^2, 2, mean)[-vx] #sigma
	  YHAT <- as.matrix(PRED)[ ,-vx]
	}

	names(sigma) <- sub(".", "", names(sigma))
	colnames(YHAT) <- sub(".", "", colnames(YHAT))
	
	return(list(dnn = fit, sigma = sigma, YHAT = YHAT))
}

cheapBootCI <- function(est, out, nboot, ...)
{
 	# Compute cheap bootsrap CI
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

#' @title SEM-based out-of-sample prediction using DNN
#'
#' @description Predict method for DNN objects.
#'
#' @param object A model fitting object from \code{SEMdnn()} function. 
#' @param newdata A matrix containing new data with rows corresponding to
#' subjects, and columns to variables. If newdata = NULL, the train data are used.
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
#' dnn0 <- SEMdnn(ig, data[train, ], algo ="layerwise",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
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
#' dnn1 <- SEMdnn(ig, data[train, ], outcome[train], algo ="layerwise",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
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
	stopifnot(inherits(object$model[[1]][[1]], "DNN"))
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
	  vx <- colnames(fit$data$X)
	  vy <- colnames(fit$data$Y)
	  pred <- extract_fitted_values(fit, Z_test)$y_hat
	  yhat <- cbind(yhat, pred)
	  yn <- c(yn, vy)
	}

	yobs <- Z_test[, yn]
	PE <- colMeans((yobs - yhat)^2)
	#PE <- ifelse(PE > 1, NA, PE)
	PE <- ifelse(PE > 1, 1, PE)
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
#' sums the products across all hidden neurons, as proposed by Olden (2002; 2004).
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
#' neural networks. Ecological Modelling, 154(1-2): 135-150.
#' https://doi.org/10.1016/S0304-3800(02)00064-9
#' 
#' Olden, Julian; Joy, Michael K; Death, Russell G (2004). An accurate comparison of methods
#' for quantifying variable importance in artificial neural networks using simulated data. 
#' Ecological Modelling, 178 (3-4): 389-397. https://doi.org/10.1016/j.ecolmodel.2004.03.013
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # Load Sachs data (pkc)
#' ig<- sachs$graph
#' data<- sachs$pkc
#' data<- log(data)
#'
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' #ncores<- parallel::detectCores(logical = FALSE)
#'
#' dnn0<- SEMdnn(ig, data[train, ], outcome = NULL, algo= "structured",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
#' 
#' cw<- getConnectionWeight(dnn0, thr = 0.3, verbose = FALSE)
#' gplot(cw$dag, l="circo")
#' table(E(cw$dag)$color)
#' }
#' }
#'
#' @export
#' 
getConnectionWeight <- function(object, thr = NULL, verbose = FALSE, ...)
{
	stopifnot(inherits(object$model[[1]][[1]], "DNN"))
	algo <- object$model[[1]][[1]]$param$algo
	bias <- object$model[[1]][[1]]$param$bias

	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}
	A <- as_adjacency_matrix(dag, sparse=FALSE) # raw:x; col:y
	
	nboot<- length(object$model)
	est_boot <- list()
	pb0 <- pb(nrep = nboot , snow = FALSE)

	for(b in 1:nboot) { #b=1
	   pb0$tick()
	   model <- object$model[[b]]
	   nrep <- length(model)
	   est <- NULL
	   for (j in 1:nrep) { #j=1
		 vx <- colnames(model[[j]]$data$X)
		 vy <- colnames(model[[j]]$data$Y)
		 W <- model[[j]]$weights[[1]] #str(W)
		 #if (algo == "autoencoder") {
		 #  h <- (length(W)-1)
		 #  #w <- W$W
		 #  W <- W[-c(1:h/2,h+1)] #str(W)
		 #}
		 #else{
		   w <- t(W[[1]])
		   if (bias) K=seq(3,length(W),by=2) else K=2:length(W)
		   for (k in K) w <- w %*% t(W[[k]])
		 #}
		 rownames(w) <- vx <- sub(".", "", vx)
		 colnames(w) <- vy <- sub(".", "", vy)
		 WA <- as.matrix(w * A[vx, vy])

		 for (k in 1:length(vy)) {
			label <- data.frame(
					  lhs = vy[k],
					  op = "~",
					  rhs = vx)
			est <- rbind(est, cbind(label, WA[vx,vy[k]]))
	     }
	   }
	   est<- est[est[,4] != 0, ]
	   rownames(est) <- NULL
	   colnames(est)[4] <- "weight"

	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame)) #dim(EST)
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)
	
	# gplot(dag0$dag)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

#' @title Gradient Weight method for neural network variable importance
#'
#' @description  The function computes the gradient matrix, i.e., the average
#' marginal effect of the input variables w.r.t the neural network model,
#' as discussed by Scholbeck et al (2024).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' gradient weights to color the graph. If thr = NULL (default), the threshold
#' is set to thr = 0.5*max(abs(gradient weights)).
#' @param verbose A logical value. If FALSE (default), the processed graph
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details The gradient weights method approximate the derivative (the gradient)
#' of each output variable (y) with respect to each input variable (x) evaluated at
#' each observation (i=1,...,n) of the training data. The contribution of each input 
#' is evaluated in terms of both magnitude taking into account not only the connection
#' weights and activation functions, but also the values of each observation of the
#' input variables. 
#' Once the gradients for each variable and observation, a summary gradient is calculated
#' by averaging over the observation units. Finally, the average weights are entered into
#' a matrix, W(pxp) and the element-wise product with the binary (1,0) adjacency matrix,
#' A(pxp) of the input DAG, W*A maps the weights on the DAG edges.
#' Note that the operations required to approsimate partial derivatives are time consuming
#' compared to other methods such as Olden's (connection weight). The computational
#' time increases with the size of the neural network or the size of the data.
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
#' Scholbeck, C.A., Casalicchio, G., Molnar, C. et al. Marginal effects for non-linear prediction
#' functions. Data Min Knowl Disc 38, 2997–3042 (2024). https://doi.org/10.1007/s10618-023-00993-x
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # Load Sachs data (pkc)
#' ig<- sachs$graph
#' data<- sachs$pkc
#' data<- log(data)
#'
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' #ncores<- parallel::detectCores(logical = FALSE)
#'
#' dnn0<- SEMdnn(ig, data[train, ], outcome = NULL, algo= "neuralgraph",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
#' 
#' gw<- getGradientWeight(dnn0, thr = 0.3, verbose = FALSE)
#' gplot(gw$dag, l="circo")
#' table(E(gw$dag)$color)
#' }
#' }
#'
#' @export
#' 
getGradientWeight<- function(object, thr = NULL, verbose = FALSE, ...)
{
	stopifnot(inherits(object$model[[1]][[1]], "DNN"))
	algo <- object$model[[1]][[1]]$param$algo
	
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}
	V(dag)$name <- paste0("z", V(dag)$name)
	pe<- igraph::as_data_frame(dag)[ ,c(1,2)]
	y <- split(pe, f=pe$to)
	
	nboot<- length(object$model)
	est_boot <- list()
	pb0 <- pb(nrep = nboot, snow = FALSE)
	
	for(b in 1:nboot) {
	  pb0$tick()
	  model<- object$model[[b]]
	  W <- list()
	  est <- NULL

	  #if (algo == "structured" | algo == "neuralgraph") {
	  if (length(model) == 1) {
	   	W <- getMarginalEffects(model[[1]], y = y)$ame
		for (k in 1:length(W)) {
		 vx <- names(W[[k]])
		 vy <- names(W)[k]
		 estk <- data.frame(
				lhs = rep(vy, length(vx)),
				op = "~",
				rhs = vx,
				grad = W[[k]])
		 est <- rbind(est, estk)
	    }
	  }
	  
	 #if (algo == "layerwise" | algo == "nodewise") {
	 if (length(model) > 1) {
		for (j in 1:length(model)) {
		  Idy <- which(names(y) %in% colnames(model[[j]]$data$Y))
		  W[[j]] <- getMarginalEffects(model[[j]], y = y[Idy])$ame
		  vy <- names(W[[j]])
		  for (k in 1:length(vy)) {
			vx <- names(W[[j]][[k]])
			estk <- data.frame( 
					 lhs = rep(vy[k], length(vx)),
					 op = "~",
					 rhs = vx,
					 grad = W[[j]][[k]])
			est <- rbind(est, estk)
		  }
		}		
	  }
	
	  rownames(est) <- NULL
	  est$lhs <- sub(".", "", est$lhs)
	  est$rhs <- sub(".", "", est$rhs)
	  	 
	  est_boot[[b]] <- est   
	}
	
	V(dag)$name <- sub(".", "", V(dag)$name)
	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

getMarginalEffects <- function(modelY, y, epsilon = 0.1, ...)
{
	X <- modelY$data$X
	# Y <- modelY$data$Y
	h <- epsilon * apply(X, 2, sd)
	fme <- list()
	ame <- list()

	for(j in 1:length(y)) {
	 vy <- names(y)[j]
	 vx <- y[[j]]$from  
	 x0 <- X
	 f <- function(x0) extract_fitted_values(modelY, x0)$y_hat[ ,vy]
	 f_x0 <- f(x0)

	 H <- as.data.frame(matrix(NA, nrow(x0), length(vx)))
	 colnames(H) <- vx
	 x0_tmp <- x0

	  for (k in 1:length(vx)) {
	 	x0_tmp[,k] <- x0_tmp[,k] + h[k]
	 	H[,k] <- ( f(x0_tmp) - f_x0 )/h[k]
	  }

	 fme[[j]] <- H
	 ame[[j]] <- apply(H, 2, mean)
	}

	names(fme) <- names(ame) <- names(y)
	
	return(list(fme = fme, ame = ame))
}

#' @title Test for the significance of neural network input nodes
#'
#' @description The function computes a formal test for the significance of
#' neural network input nodes, based on a linear relationship between the
#' observed output and the predicted values of an input variable, when all
#' other input variables are maintained at their mean (or zero) values, as
#' proposed by Mohammadi (2018).
#'
#' @param object A neural network object from \code{SEMdnn()} function. 
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' t-test values to color the graph. If thr = NULL (default), the threshold
#' is set to thr = 0.5*max(abs(t-test values)).
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details A neural network is trained, taking into account the number of
#' hidden layers, neurons, and activation function. Then, network's output is
#' simulated to get  the predicted values of the output variable, fixing all the
#' inputs (with the exception of one nonconstant input variable) at their mean
#' values. Subsequently, the network's predictions are stored after this process
#' is completed for each input variable. As last step, multiple regression analysis
#' is applied node-wise (mapping the input DAG) on the observed output nodes with
#' the predicted values of the input nodes as explanatory variables. The statistical
#' significance of the coefficients is evaluated using standard t-student values,
#' which represent the importance of the input variables. 
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
#' S. Mohammadi. A new test for the significance of neural network inputs.
#' Neurocomputing 2018; 273: 304-322. https://doi.org/10.1016/j.neucom.2017.08.007
#'
#' @examples
#'
#' \donttest{
#' if (torch::torch_is_installed()){
#'
#' # Load Sachs data (pkc)
#' ig<- sachs$graph
#' data<- sachs$pkc
#' data<- log(data)
#'
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' #ncores<- parallel::detectCores(logical = FALSE)
#'
#' dnn0<- SEMdnn(ig, data[train, ], outcome = NULL, algo= "nodewise",
#' 			hidden = c(10,10,10), link = "selu", bias = TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
#' 
#' st<- getSignificanceTest(dnn0, thr = NULL, verbose = FALSE)
#' gplot(st$dag, l="circo")
#' table(E(st$dag)$color)
#' }
#' }
#'
#' @export
#' 
getSignificanceTest <- function(object, thr = NULL, verbose = FALSE, ...)
{
	stopifnot(inherits(object$model[[1]][[1]], "DNN"))
	algo <- object$model[[1]][[1]]$param$algo
	
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}
	V(dag)$name <- paste0("z", V(dag)$name)
	pe<- igraph::as_data_frame(dag)[ ,c(1,2)]
	y <- split(pe, f=pe$to)

	nboot<- length(object$model)
	est_boot <- list()
	pb0 <- pb(nrep = nboot, snow = FALSE)
	
	for(b in 1:nboot) {
	  pb0$tick()
	  model<- object$model[[b]]
	  W <- list()
	  est <- NULL
	
	  #if (algo == "structured" | algo == "neuralgraph") {
	  if (length(model) == 1) {
	 	W <- getTestValues(model[[1]], y = y)
		for (k in 1:length(W)) {
		 vx <- rownames(W[[k]])
		 vy <- names(y)[k]
		 estk <- data.frame(
				lhs = rep(vy, length(vx)),
				op = "~",
				rhs = vx,
				t_test = W[[k]][,3])
		 est <- rbind(est, estk)
	    }
	  }
	   
	  #if (algo == "layerwise" | algo == "nodewise") {
	  if (length(model) > 1) {
		for (j in 1:length(model)) {
		  Idy <- which(names(y) %in% colnames(model[[j]]$data$Y))
		  W[[j]] <- getTestValues(model[[j]], y = y[Idy])
		  vy <- names(W[[j]])
		  for (k in 1:length(vy)) {
			vx <- rownames(W[[j]][[k]])
			estk <- data.frame( 
					 lhs = rep(vy[k], length(vx)),
					 op = "~",
					 rhs = vx,
					 t_test = W[[j]][[k]][,3])
			est <- rbind(est, estk)
		  }
		}
	  }

	  rownames(est) <- NULL
	  est$lhs <- sub(".", "", est$lhs)
	  est$rhs <- sub(".", "", est$rhs)
	 		
	  est_boot[[b]] <- est
	}

	V(dag)$name <- sub(".", "", V(dag)$name)
	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

getTestValues <- function(modelY, y, ...)
{
	X <- modelY$data$X
	Y <- modelY$data$Y
	est <- list()

	for(j in 1:length(y)) {
	 vy <- names(y)[j]
	 vx <- y[[j]]$from  
	 
	 X0 <- as.data.frame(matrix(0, nrow(X), ncol(X)))
	 colnames(X0) <- colnames(X)
	 Yhatj <- NULL
	 
	 for (k in 1:length(vx)) {
	  X0[ ,k] <- X[ ,k]
	  Yhatk <- extract_fitted_values(modelY, X)$y_hat[ ,vy]
	  Yhatj <- cbind(Yhatj, Yhatk)
	 } 
	 colnames(Yhatj) <- vx
	 
	 # Fit a multiple linear model of Yj on Yhatj
	 if (ncol(Yhatj) >= nrow(Yhatj)) {
	  set.seed(1)
	  idx <- sample(1:ncol(Yhatj), round(0.9*nrow(Yhatj)))
	  Yhatj <- Yhatj[ ,idx]
	 }
	 Yj <- data.frame(Y[ ,vy], Yhatj)
	 colnames(Yj)[1] <- vy
	 f <- paste0(vy, "~.")
	 fit <- lm(eval(f), data = Yj)
	 estj <- data.frame(summary(fit)$coefficients)[-1, ]
	 colnames(estj) <- c("est", "se", "t", "pvalue")
	 estj$t <- ifelse(abs(estj$t) < 3, estj$t, sign(estj$t)*3)
	 
	 est[[j]] <- estj
	} 
	names(est) <- names(y)
	
	return(est)
}
