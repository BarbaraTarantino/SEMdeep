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

#' @title SEM-based out-of-sample prediction using layer-wise ordering
#'
#' @description Given the values of (observed) x-variables in a SEM,
#' this function may be used to predict the values of (observed) y-variables.
#' The predictive procedure consists of two steps. First, the topological layer
#' ordering of the input graph is defined. Then, the node y values in a layer are
#' predicted, where the nodes in successive layers act as x-predictors. 
#' 
#' @param object An object, as that created by the function \code{SEMrun()}
#' with the argument \code{group} set to the default \code{group = NULL}.
#' @param newdata A matrix with new data, with rows corresponding to subjects,
#' and columns to variables. 
#' @param newoutcome A new character vector (as.factor) of labels for a categorical
#' output (target)(default = NULL).
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details  The function first creates a layer-based structure of the
#' input graph. Then, a SEM-based predictive approach (Rooij et al., 2022) 
#' is used to produce predictions while accounting for the graph structure
#' based on the topological layer (j=1,…,L) of the input graph. In each iteration,
#' the response (output) variables, y are the nodes in the j=1,...,(L-1) layer and
#' the predictor (input) variables, x are the nodes belonging to the successive,
#' (j+1),...,L layers.
#' Predictions (for y given x) are based on the (joint y and x) model-implied 
#' variance-covariance (Sigma) matrix and mean vector (Mu) of the fitted SEM,
#' and the standard expression for the conditional mean of a multivariate normal
#' distribution. Thus, the layer structure described in the SEM is taken into
#' consideration, which differs from ordinary least squares (OLS) regression.
#'
#' @return A list of 3 objects:
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
#' @references 
#' 
#' de Rooij M, Karch JD, Fokkema M, Bakk Z, Pratiwi BC, and Kelderman H
#' (2023). SEM-Based Out-of-Sample Predictions, Structural Equation Modeling:
#' A Multidisciplinary Journal, 30:1, 132-148. <https://doi.org/10.1080/10705511.2022.2061494>
#'
#' Grassi M, Palluzzi F, Tarantino B (2022). SEMgraph: An R Package for Causal Network
#' Analysis of High-Throughput Data with Structural Equation Models.
#' Bioinformatics, 38 (20), 4829–4830. <https://doi.org/10.1093/bioinformatics/btac567>
#'
#' Grassi, M., Tarantino, B. (2025). SEMdag: Fast learning of Directed Acyclic Graphs via
#' node or layer ordering. PLoS ONE 20(1): e0317283. https://doi.org/10.1371/journal.pone.0317283
#'
#' @examples
#'
#' # load ALS data
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' group<- alsData$group
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' # predictors, source+mediator; outcomes, mediator+sink
#'
#' ig <- alsData$graph; gplot(ig)
#' sem0 <- SEMrun(ig, data[train,], algo="ricf", n_rep=0)
#' pred0 <- predict(sem0, newdata=data[-train,], verbose=TRUE) 
#' 
#' # predictors, source+mediator+group; outcomes, source+mediator+sink
#'
#' ig1 <- mapGraph(ig, type = "group"); gplot(ig1)
#' data1 <- cbind(group, data); head(data1[,5])
#' sem1 <- SEMrun(ig1, data1[train,], algo="ricf", n_rep=0)
#' pred1 <- predict(sem1, newdata= data1[-train,], verbose=TRUE) 
#'
#' # predictors, source nodes; outcomes, sink nodes
#'
#' ig2 <- mapGraph(ig, type = "source"); gplot(ig2)
#' sem2 <- SEMrun(ig2, data[train,], algo="ricf", n_rep=0)
#' pred2 <- predict(sem2, newdata=data[-train,], verbose=TRUE)
#'
#' @method predict SEM
#' @export
#' @export predict.SEM
#' 
predict.SEM <- function(object, newdata, newoutcome = NULL, verbose = FALSE, ...)
{
	# set data and graph objects
	stopifnot(inherits(object, "SEM"))
	fit<- object$fit
	graph<- object$graph
	data<- object$data[,-1]
	if (!is.null(newoutcome)) {
	 out<- model.matrix(~newoutcome-1)
	 colnames(out)<- gsub("newoutcome", "", colnames(out))
	 newdata<- cbind(out, newdata)
	 graph<- mapGraph(graph, type="outcome", C=ncol(out))
	 V(graph)$name[igraph::degree(graph, mode="out") == 0]<- colnames(out)
	}
	graph<- graph2dag(graph, data, bap=FALSE)
	L<- buildLevels(graph)
	
	# SEM predition on test data
	mp<- apply(data, 2, mean)
	sp<- apply(data, 2, sd)
	yobs<- NULL
	yhat<- NULL

	if (inherits(fit, "lavaan")) {
	  #Sigma<- lavaan::fitted(fit)$cov
	  Sigma<- lavaan::lavInspect(fit, "sigma")
	  colnames(Sigma)<- sub(".", "", colnames(Sigma))
	  rownames(Sigma)<- colnames(Sigma)
	  #mu<- fitted(fit$fit)$mean
	}else{
	  Sigma<- fit$Sigma
	  #mu<- rep(0, p)
	}

	for (l in 1:(length(L)-1)) {
	  yn<- L[[l]]
	  xn<- unlist(L[(l+1):length(L)])
	  Sxx<- Sigma[xn, xn]
	  Sxy<- Sigma[xn, yn]
	  mx<- rep(0, length(xn))
	  my<- rep(0, length(yn))
	  xtest<- as.matrix(newdata[, xn])
	  xtest<- scale(xtest, center=mp[xn], scale=sp[xn])
	  #xtest<- scale(xtest, center = mx, scale = TRUE)
	  n<- nrow(xtest)
	  py<- length(yn)
	  My<- matrix(my, n, py, byrow = TRUE)
	  if (corpcor::is.positive.definite(Sxx)) {
	  	yhatl<- My + xtest %*% solve(Sxx) %*% Sxy 
	  }else{
	  	yhatl<- My + xtest %*% Sxy
	  }
	  yobsl<- newdata[, yn]
	  yobsl<- scale(yobsl, center=mp[yn], scale=sp[yn])
	  #yobsl<- scale(yobsl) #dim(yhatl)
	  colnames(yobsl)<- colnames(yhatl)<- yn
	  yobs<- cbind(yobs, yobsl)
	  yhat<- cbind(yhat, yhatl)
	}

	PE<- colMeans((yobs - yhat)^2)
	#PE <- ifelse(PE > 1, NA, PE)
	PE <- ifelse(PE > 1, 1, PE)
	pe<- mean(PE, na.rm = TRUE)
			
	Shat<- Sigma
	V<- colnames(Shat)
	Sobs<- cor(newdata[, V])
	E<- Sobs - Shat
	SRMR<- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	
	if (verbose) print(c(amse=pe, r2=1-pe, srmr=SRMR))
	
	return(list(PE=c(amse=pe, r2=1-pe, srmr=SRMR), mse=PE, Yhat=yhat))
}

#' @title Create a plot for a neural network model
#'
#' @description The function uses the \code{\link[NeuralNetTools]{plotnet}}
#' function of the \pkg{NeuralNetTools} R package to draw a neural network
#' plot and visualize the hidden layer structure. 
#'
#' @param object A neural network model object
#' @param hidden The hidden structure of the object
#' @param bias A logical value, indicating whether to draw biases in 
#' the layers (default = FALSE).
#' @param sleep Suspend plot display for a specified time (in secs, default = 2).
#' @param ... Currently ignored.
#'
#' @details The induced subgraph of the input graph mapped on data 
#' variables. Based on the estimated connection weights, if the connection
#' weight W > 0, the connection is activated and it is highlighted in red;  
#' if W < 0, the connection is inhibited and it is highlighted in blue.  
#' 
#' @return The function invisibly returns the graphical objects representing
#' the neural network architecture designed by NeuralNetTools. 
#' 
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#' 
#' Beck, M.W. 2018. NeuralNetTools: Visualization and Analysis Tools for Neural 
#' Networks. Journal of Statistical Software. 85(11):1-20.
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
#' dnn0 <- SEMdnn(ig, data, train=1:nrow(data), algo = "layerwise",
#' 			hidden = c(10, 10, 10), link = "selu", bias =TRUE,
#' 			epochs = 32, patience = 10, verbose = TRUE)
#'
#'  #Visualize the neural networks per each layer of dnn0
#'  nplot(dnn0, hidden = c(10, 10, 10), bias = FALSE)
#' }
#' }
#'
#' @export
#' 
nplot <- function (object, hidden, bias = TRUE, sleep = 2, ...) 
{
    stopifnot(inherits(object$model[[1]][[1]], "DNN"))
	algo <- object$model[[1]][[1]]$param$algo
	#if (algo == "autoencoder") stop("algo = \"", algo, "\" is not supported!")
	model <- object$model[[1]]
	g <- list()
	
    for (j in 1:length(model)) {
        W <- model[[j]]$weights[[1]]
        input <- ncol(W[[1]])
        if (algo == "structured") {
		 if (hidden[1] < input) hidden <- hidden + input #n_hid > n_inp 
		}
		output <- nrow(W[[length(W) - 1]])
        w0 <- list()
        for (i in seq(2, length(W), by = 2)) {
            wi <- cbind(W[[i]], W[[i - 1]])
            w0 <- c(w0, list(t(wi)))
        }
        w1 <- unlist(w0)
        struct1 <- c(input, hidden, output)
        g[[j]] <- NeuralNetTools::plotnet(w1, struct1, pos_col = "red", 
            neg_col = "blue", bias = bias)
        Sys.sleep(sleep)
    }
	
    return(invisible(g))
}

#' @title Map additional variables (nodes) to a graph object
#'
#' @description The function insert additional nodes to a graph object.
#' Among the node types, additional source or sink nodes can be added. 
#' Source nodes can represent: (i) data variables; (ii) a group variable;
#' (iii) latent variables (LV). Vice versa, sink nodes represent the levels
#' of a categorical outcome variable and are linked with all graph nodes.
#' Moreover, \code{mapGraph()} can also create a new graph object starting
#' from a compact symbolic formula. 
#' 
#' @param graph An igraph object.
#' @param type A character value specifying the type of mapping. Five 
#' types can be specified.
#' \enumerate{
#' \item "source", source nodes are linked to sink nodes of the graph.
#' \item "group", an additional group source node is added to the graph.
#' \item "outcome", additional c=1,2,...,C sink nodes are added to the graph.
#' \item "LV", additional latent variable (LV) source nodes are added to the graph. 
#' \item "clusterLV", a series of clusters for the data are computed
#' and a different LV source node is added separately for each cluster.
#' }
#' @param C the number of labels of the categorical sink node (default = NULL).
#' @param LV The number of LV source nodes to add to the graph. This argument 
#' needs to be specified when \code{type = "LV"}. When \code{type = "clusterLV"}
#' the LV number is defined internally equal to the number of clusters.
#' (default = NULL).
#' @param f A formula object (default = NULL). A new graph object is created
#' according to the specified formula object.
#' @param verbose If TRUE disply the mapped graph (default = FALSE) 
#' @param ... Currently ignored.
#'
#' @return mapGraph returns invisibly the graphical object with the
#' mapped node variables.
#' 
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @examples
#' 
#' # Load Amyotrophic Lateral Sclerosis (ALS)
#' ig<- alsData$graph; gplot(ig)
#' 
#' # ... map source nodes to sink nodes of ALS graph 
#' ig1 <- mapGraph(ig, type = "source"); gplot(ig1, l="dot")
#' 
#' # ... map group source node to ALS graph 
#' ig2 <- mapGraph(ig, type = "group"); gplot(ig2, l="fdp")
#' 
#' # ... map outcome sink (C=2) to ALS graph 
#' ig3 <- mapGraph(ig, type = "outcome", C=2); gplot(ig3, l="fdp")
#' 
#' # ... map LV source nodes to ALS graph 
#' ig4 <- mapGraph(ig, type = "LV", LV = 3); gplot(ig4, l="fdp")
#' 
#' # ... map LV source nodes to the cluster nodes of ALS graph 
#' ig5 <- mapGraph(ig, type = "clusterLV"); gplot(ig5, l="dot")
#'
#' # ... create a new graph with the formula variables
#' formula <- as.formula("z4747 ~ z1432 + z5603 + z5630")
#' ig6 <- mapGraph(f=formula); gplot(ig6)
#
#' @export
#' 
mapGraph <- function(graph, type, C = NULL, LV = NULL, f = NULL, verbose = FALSE, ...)
{
	if (!is.null(f)) {
	  g <- map_formula(f, verbose=verbose)
	  return(invisible(g))
	}
	if (type == "source") g <- map_source(graph, verbose=verbose)
	if (type == "group") g <- map_group(graph, verbose=verbose)
	if (type == "outcome") g <- map_outcome(graph, C=C, verbose=verbose)
	if (type == "LV") g <- map_LV(graph, LV=LV, cg=NULL, verbose=verbose)
	if (type == "clusterLV"){
	  cg<- SEMgraph::clusterGraph(graph, size=0)
	  LV<- length(table(cg))
	  g<- map_LV(graph, LV=LV, cg=cg, verbose=verbose)
	}
	return(invisible(g))
}

map_formula <- function(f, verbose, ...)
{
	vf <- all.vars(f)
	y <- vf[1]
	xn <- vf[-1]
	gout <- make_empty_graph(length(vf))
	V(gout)$name <- c(vf)
	E <- NULL
	for(k in 1:length(xn)){
	 E <- c(E, xn[k], y)
	}
	gout <- gout + igraph::edges(E)
	if (verbose) gplot(gout)
	
	return(gout) 
}

map_source <- function(graph, verbose, ...)
{
	din <- igraph::degree(graph, mode= "in")
	dout <- igraph::degree(graph, mode = "out")
	xn <- V(graph)$name[din == 0]
	yn <- V(graph)$name[dout == 0]
	gout <- make_empty_graph(length(c(xn,yn)))
	V(gout)$name <- c(xn,yn)
	E <- NULL
	 for(k in 1:length(xn)){
	  for(j in 1:length(yn)){
		E <- c(E, xn[k], yn[j])
	  }
	 }
	gout <- gout + igraph::edges(E)  
	if (verbose) gplot(gout)
	
	return(gout)	
}

map_group <- function(graph, verbose, ...)
{
	gout <- graph + igraph::vertices("group")
	nodes<- V(graph)$name
	E <- NULL
	 for(v in 1:length(nodes)){
	  	E <- c(E, "group", nodes[v])
	  }
	gout <- gout + igraph::edges(E)
	V(gout)$color[V(gout)$name == "group"] <- "green"
	if (verbose) gplot(gout)
	
	return(gout)	
}

map_outcome <- function(graph, C, verbose, ...)
{
	outcome<- paste0("out", 1:C)
	gout <- graph + igraph::vertices(outcome)
	nodes<- V(graph)$name
	E <- NULL
	 for(v in 1:length(nodes)){
	  	for (k in 1:C) E <- c(E, nodes[v], paste0("out",k))
	  }
	gout <- gout + igraph::edges(E)
	V(gout)$color[V(gout)$name %in% paste0("out", 1:C)] <- "yellow"
	if (verbose) gplot(gout)
	
	return(gout)	
}

map_sink <- function(graph, C, verbose, ...)
{
	outcome<- paste0("out", 1:C)
	gout <- graph + igraph::vertices(outcome)
	dout<- igraph::degree(graph, mode = "out")
	leaf<- V(graph)$name[dout == 0]
	E <- NULL
	 for(v in 1:length(leaf)){
	  	for (k in 1:C) E <- c(E, leaf[v], paste0("out",k))
	  }
	gout <- gout + igraph::edges(E)
	V(gout)$color[V(gout)$name %in% paste0("out", 1:C)] <- "yellow"
	if (verbose) gplot(gout)
	
	return(gout)	
}

map_LV <- function(graph, LV, cg, verbose, ...)
{
	VH <- paste0("LV", 1:LV)
	gH <- graph + igraph::vertices(VH)
	E <- NULL
	if (is.null(cg)) {
	 for(v in 1:length(VH)){
	   for(i in 1:vcount(graph)){ #i=1
		 E <- c(E, VH[v], V(graph)$name[i])
	   }
	 }
	}
	if (!is.null(cg)) {
	 VH <- paste0("LV", cg)
	 for(i in 1:vcount(graph)){ #i=1
	   E <- c(E, VH[i], names(cg[i]))
	 }
	}
	gH <- gH + igraph::edges(E)
	V(gH)$color[V(gH)$name %in% VH] <- "green"
	if (verbose) gplot(gH, l="fdp")
	
	return(gH)	
}

buildLevels <- function(dag, ...)
{
	Leaf_removal <- function(dag)
	{
	 levels <- list()
	 level <- 1
	 repeat {
	  leaves <- igraph::degree(dag, mode= "out")
	  levels[[level]]<- names(leaves)[leaves == 0]
  	  dag <- delete_vertices(dag, names(leaves)[leaves == 0])
	  level <- level+1
	  if (vcount(dag)==0 | ecount(dag)==0) break
	 }
	 levels[[level]] <- V(dag)$name
	 names(levels)<- 1:level
	 return(levels)
	}

	# leaf-removal(dag)
	l1<- Leaf_removal(dag)
	if (length(l1) == 2) return(l1)
	# leaf removal(dagT)
	adj <- as_adjacency_matrix(dag, sparse=FALSE)
	dagT <- graph_from_adjacency_matrix(t(adj), mode="directed")
	l2 <- Leaf_removal(dagT)
	l2 <- rev(l2)
	# number-of-layers 
	L <- max(length(l1), length(l2))

	# combine BU-ordering (dag+dagT)
	l3 <- list()
	l3[[1]] <- l1[[1]] #sink
	l3[[L]] <- l2[[L]] #source
	for (k in 2:(L-1)){
	 lk <- unique(c(l1[[k]], l2[[k]]))
	 Lk <- unlist(l3[c(1:(k-1),L)])
	 l3[[k]] <- setdiff(lk, Lk)
	}

	return(l3)
}

#' @title Prediction evaluation report of a classification model
#'
#' @description This function builds a report showing the main classification 
#' metrics. It provides an overview of key evaluation metrics like precision, 
#' recall, F1-score, accuracy, Matthew's correlation coefficient (mcc) and
#' support (testing size) for each class in the dataset and averages (macro or
#' weighted) for all classes.
#' 
#' @param yobs A vector with the true target variable values. 
#' @param yhat A matrix with the predicted target variables values. 
#' @param CM An optional (external) confusion matrix CxC. 
#' @param verbose A logical value (default = FALSE). If TRUE, the confusion
#' matrix is printed on the screen, and if C=2, the density plots of the
#' predicted probability for each group are also printed.
#' @param ... Currently ignored.
#'
#' @details Given one vector with the true target variable labels, 
#' and the a matrix with the predicted target variable values for each class, 
#' a series of classification metrics is computed. 
#' For example, suppose a 2x2 table with notation
#'
#' \tabular{rcc}{ \tab Predicted \tab \cr Observed \tab Yes Event \tab No Event
#' \cr Yes Event \tab A \tab C \cr No Event \tab B \tab D \cr }
#'
#' The formulas used here for the label = "Yes Event" are:
#'
#' \deqn{pre = A/(A+B)} \deqn{rec = A/(A+C)} 
#' \deqn{F1 = (2*pre*rec)/(pre+rec)}
#' \deqn{acc = (A+D)/(A+B+C+D)}
#' \deqn{mcc = (A*D-B*C)/sqrt((A+B)*(C+D)*(A+C)*(B+D))}
#' 
#' Metrics analogous to those described above are calculated for the label
#' "No Event", and the weighted average (averaging the support-weighted mean
#' per label) and macro average (averaging the unweighted mean per label) are
#' also provided.
#'
#' @return A list of 3 objects:
#' \enumerate{
#' \item "CM", the confusion matrix between observed and predicted counts.
#' \item "stats", a data.frame with the classification evaluation statistics.
#' \item "cls", a data.frame with the predicted probabilities, predicted
#' labels and true labels of the categorical target variable.
#' }
#'
#' @author Barbara Tarantino \email{barbara.tarantino@unipv.it}
#'
#' @references 
#' 
#' Sammut, C. & Webb, G. I. (eds.) (2017). Encyclopedia of Machine Learning 
#' and Data Mining. New York: Springer. ISBN: 978-1-4899-7685-7 
#' 
#' @examples
#'
#' \donttest{
#' # Load Sachs data (pkc)
#' ig<- sachs$graph
#' data<- sachs$pkc
#' data<- transformData(data)$data
#' group<- sachs$group
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' 
#' #...with a categorical (as.factor) variable (C=2)
#' outcome<- factor(ifelse(group == 0, "control", "case"))
#' res<- SEMml(ig, data[train, ], outcome[train], algo="rf")
#' pred<- predict(res, data[-train, ], outcome[-train], verbose=TRUE)
#' 
#' yobs<- outcome[-train]
#' yhat<- pred$Yhat[ ,levels(outcome)]
#' cls<- classificationReport(yobs, yhat)
#' cls$CM
#' cls$stats
#' head(cls$cls)
#'
#' #...with predicted probabiliy density plots, if C=2
#' cls<- classificationReport(yobs, yhat, verbose=TRUE)
#' 
#' #...with a categorical (as.factor) variable (C=3)
#' group[1:400]<- 2; table(group)
#' outcome<- factor(ifelse(group == 0, "control",
#'					ifelse(group == 1, "case1", "case2")))
#' res<- SEMml(ig, data[train, ], outcome[train], algo="rf")
#' pred<- predict(res, data[-train, ], outcome[-train], verbose=TRUE)
#'
#' yobs<- outcome[-train]
#' yhat<- pred$Yhat[ ,levels(outcome)]
#' cls<- classificationReport(yobs, yhat)
#' cls$CM
#' cls$stats
#' head(cls$cls)
#' }
#'
#' @export
#'
classificationReport<- function(yobs, yhat, CM=NULL, verbose = FALSE, ...)
{  
	# CxC Confusion Matrix
	if (inherits(CM, "table")){
		cm <- CM
	}else{ 
		expit <- apply(yhat, 2, function(x) exp(x)/(1+exp(x)))
		prob <- apply(expit, 2, function(x) x/apply(expit,1,sum))
		labels <- colnames(prob)
		pred <- apply(prob, 1, function(x) labels[which.max(x)])
		cm <- table(yobs, pred)
		if (nrow(cm) != ncol(cm)){
			message("confusion table RxC is not a confusion matrix CxC!")
			return(list(CM=matrix(0, nrow=nrow(cm), ncol=nrow(cm))))
		}
	}
	if (verbose) {
		print(cm)
		message()
	}

	n <- sum(cm) # number of observations
	nc <- nrow(cm) # number of classes
	correct <- diag(cm) # number of correctly classified obs per class 
	x <- rowSums(cm) # number of obs per class
	y <- colSums(cm) # number of predictions per class
	support <- x # n actual observation per class
	support_prop <- x / n # n actual observation per class over total 

	# Per-class precision recall f1
	precision <- correct / y 
	recall <- correct / x 
	f1 <- 2 * precision * recall / (precision + recall)
	accuracy <- sum(correct) / n

	# Macro-averaged Metrics
	macroPrecision <- mean(precision, na.rm=TRUE)
	macroRecall <- mean(recall, na.rm=TRUE)
	macroF1 <- mean(f1, na.rm=TRUE)

	# Weighted-averaged Metrics
	weightPrecision <- sum(diag(outer(precision,support_prop)))
	weightRecall <- sum(diag(outer(recall,support_prop)))
	weightF1 <- sum(diag(outer(f1,support_prop)))

	# Multiclass Matthew's Correlation Coefficient
	cov_x_y <- sum(correct) * n - sum(diag(outer(x, y)))
	cov_y_y <- n * n - sum(diag(outer(y, y)))
	cov_x_x <- n * n - sum(diag(outer(x, x)))
	denom <- sqrt(cov_x_x * cov_y_y)
	denom <- ifelse(denom == 0, 1, denom)
	mcc <- cov_x_y / denom

	res1 <- data.frame(precision, recall, f1, accuracy, mcc, support, support_prop)
	res2 <- data.frame(precision=c(macroPrecision, weightPrecision),
						recall=c(macroRecall, weightRecall),
						f1=c(macroF1, weightF1),
						accuracy=c(accuracy, accuracy),
						mcc=c(mcc,mcc),
						support=c(rep(sum(support),2)),
						support_prop=c(rep(1,2)))
	rownames(res2) <- c("macro avg", "weighted avg")
	res <- rbind(res1,res2)

	if (inherits(CM, "table")) {
		cls <- NULL
	}else{
		cls <- data.frame(prob, pred, yobs)
		if (verbose == TRUE & nrow(res1) == 2) {
		 dplot(yobs, prob, 0.5, res1$recall)
		}
	}

	return(list(CM = cm, stats = res, cls = cls))
}

dplot<- function(yobs, yhat, thr, rec, ...)
{ 
	# density plot of yhat per yobs(1,2)
	label <- levels(yobs)
	cls <- ifelse(yobs == label[1], 1, 2)
	err <- c(paste0("recall = ", round(rec[1],3)),
			 paste0("recall = ", round(rec[2],3)))
	xlim <- c(0,1)
	old.par <- par(no.readonly = TRUE)
	on.exit(par(old.par))
	par(mfrow=c(2,1), mar=rep(4.5,4))
	for (c in 1:2) {
		d <- density(yhat[cls == c, c])
		x <- d$x
		y <- d$y/max(d$y)
		plot(x, y, type = "l", xlim = xlim,
			xlab = paste0("probability (Y=", label[c],")"),
			main = list(paste0(label[c], " (", err[c], ")"),
			       cex = 1.3, col = "black", font = 2))
		polygon(x, y, col="gray", border="gray")
		#if (c == 1) {
		# region.x <- x[thr <= x & x <= xlim[2]]
		# region.y <- y[thr <= x & x <= xlim[2]]
		#}else{
		 region.x <- x[xlim[1] <= x & x <= thr]
		 region.y <- y[xlim[1] <= x & x <= thr]
		#}
		region.x <- c(region.x[1], region.x, tail(region.x,1))
		region.y <- c(0, region.y, 0)
		polygon(region.x, region.y, density=-1, col="red") 
	}
}

#' @title Cross-validation of linear SEM, ML or DNN training models
#'
#' @description The function does a R-repeated K-fold cross-validation
#' of \code{SEMrun()}, \code{SEMml()} or \code{SEMdnn()} models. 
#'
#' @param models A named list of model fitting objects from \code{SEMrun()},
#' \code{SEMml()} or \code{SEMdnn()} function, with default group=NULL (for
#' \code{SEMrun()} or outcome=NULL (for \code{SEMml()} or \code{SEMdnn()}).
#' @param outcome A character vector (as.factor) of labels for a categorical
#' output (target). If NULL (default), the categorical output (target) will
#' not be considered.
#' @param K A numerical value indicating the number of k-fold to create. 
#' @param R A numerical value indicating the number of repetitions for the k-fold
#' cross-validation. 
#' @param metric A character value indicating the metric for boxplots display, i.e.:
#' "amse", "r2", or "srmr", for continuous outcomes, and "f1", "accuracy" or "mcc",
#' for a categorical outcome (default = NULL).
#' @param ncores Number of cpu cores (default = 2).
#' @param verbose Output to console boxplots and summarized results (default = FALSE).
#' @param ... Currently ignored.
#'
#' @details Easy-to-use model comparison and selection of SEM, ML or DNN models,
#' in which several models are defined and compared in a R-repeated K-fold
#' cross-validation procedure. The winner model is selected by reporting the mean
#' predicted performances across all runs, as outline in de Rooij & Weeda (2020).
#'
#' @return A list of 2 objects: (1) "stats", a list with performance evaluation metrics.
#' If \code{outcome=FALSE}, mean and (0.025;0.0975)-quantiles of amse, r2, and srmr
#' across folds and repetitions are reported; if \code{outcome=TRUE}, mean and
#' (0.025;0.0975)-quantiles of f1, accuracy and mcc from confusion matrix averaged across
#' all repetitions are reported; and (2) "PE", a data.frame of repeated cross-validation
#' results.
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references 
#' 
#' de Rooij M, Weeda W. Cross-Validation: A Method Every Psychologist Should Know.
#' Advances in Methods and Practices in Psychological Science. 2020;3(2):248-263.
#' doi:10.1177/2515245919898466
#'
#' @examples
#'
#' \donttest{
#' # Load Amyotrophic Lateral Sclerosis (ALS)
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#' group<- alsData$group
#' 
#' # ... with continuous outcomes 
#'
#' res1 <- SEMml(ig, data, algo="tree")
#' res2 <- SEMml(ig, data, algo="rf")
#' res3 <- SEMml(ig, data, algo="xgb")
#' res4 <- SEMml(ig, data, algo="sem")
#' 
#' models <- list(res1,res2,res3,res4)
#' names(models) <- c("tree","rf","xgb","sem")
#' 
#' res.cv1 <- crossValidation(models, outcome=NULL, K=5, R=10)
#' print(res.cv1$stats)
#' 
#' #... with a categorical (as.factor) outcome
#' 
#' outcome <- factor(ifelse(group == 0, "control", "case"))
#' res.cv2 <- crossValidation(models, outcome=outcome, K=5, R=10)
#' print(res.cv2$stats)
#' }
#'
#' @export
#'
crossValidation <- function(models, outcome=NULL, K=5, R=1, metric=NULL, ncores=2, verbose=FALSE, ...)
{ 
	message('Running Cross-validation...')
	# CV without repeats
	if (R == 1) {
	 resk <- crossValidationR1(models, outcome, K, ncores, seed=NULL)
	 if (verbose) print(round(resk$stats,3))
	 return(resk)
	}
	# CV with repeats
	resr<- list()
	for (r in 1:R) {
	 cat("r-repeat =", r, "\n")
	 resr[[r]] <- crossValidationR1(models, outcome, K, ncores, seed=NULL)$stats
	}
	
	# Parallel loop of repeats
	#require(foreach)
	#cl <- parallel::makeCluster(ncores)
	#doSNOW::registerDoSNOW(cl)
	#opts <- list(progress = pb(nrep = R, snow = TRUE))
	#resr <- foreach(r=1:R, .options.snow=opts) %dopar% {
    # SEMdeep:::crossValidationR1(models, outcome, K, ncores, seed=NULL)$stats
 	#}
	#parallel::stopCluster(cl)

	RESR<- do.call(rbind, lapply(resr, as.data.frame))
	rownames(RESR) <- NULL
	M <- length(models)
	Repeats <- rep(1:R, each=M)
	Models <- rep(1:M, times=R)
	PE <- cbind(Repeats, Models, RESR)
	
	# make output list
	metrics <- colnames(PE)[3:5]
	OUT <- list()
	for (i in 1:3) {
	 out <- NULL
	 winmat <- matrix(0, ncol = M, nrow = R)
		 
	 for (r in 1:R) {
	  pe <- PE[PE$Repeats == r, i+2]
	   if (metrics[i] == "amse" | metrics[i] == "srmr") {
	    winmat[r, which.min(pe)] <- 1
	   } else {
	    winmat[r ,which.max(pe)] <- 1
	   }
	 }
	 Wins <- apply(winmat,2,sum)
	
	 for (k in 1:M) {
	  pe <- PE[PE$Models == k, i+2]
	  q025 <- round(quantile(pe, .025, na.rm = TRUE),3)
	  Mean <- round(mean(pe, na.rm = TRUE),3) 
	  q975 <- round(quantile(pe, .975, na.rm = TRUE),3)
	  out<- rbind(out, cbind(q025, Mean, q975))
	 }
	 rownames(out) <- names(models)
	 colnames(out) <- c("2.5%", "mean", "97.5%")
	 
	 OUT[[i]]<- cbind(Wins, out)
	}
	names(OUT) <- metrics
	#if (verbose) print(OUT)
	
	#make boxplot
	if (verbose & !is.null(metric)) {
	 # basic boxplot
	 x <- factor(PE$Models, labels = names(models))
	 y <- PE[, which(colnames(PE) == metric)]
	 wins <- OUT[which(names(OUT) == metric)][[1]][,1]
	 boxplot(y~x, ylab=metric, xlab="Model", col=terrain.colors(4))
	 # add data points
	 for (i in 1:M) {
	  level <- levels(x)[i]
	  value <- y[x == level]
	  myjitter <- jitter(rep(i, R), amount = 1/(2*M))
	  points(myjitter, value, pch=1, col=rgb(0,0,0,0.9))
	 }
	 axis(3, seq(1, M, length.out = M), wins)
	 mtext("Number of Wins", 3, 3)
	}

	return(list(stats = OUT, PE = PE))
}

crossValidationR1 <- function(models, outcome, K, ncores, seed = NULL, ...)
{
	#set seeds and folds
	if (is.null(seed)) seed <- runif(1,0,1000)
	set.seed(seed)
	#create folds
	C <- length(levels(outcome))
	n <- nrow(models[[1]]$data)
	if (C == 0) {
	  idx <- createFolds(y=rnorm(n), k=K) #N.B. list of test indices!
	}else{
	  idx <-  createFolds(y=outcome, k=K) #N.B. list of test indices!
	}

	   # Initialise M-models loop 
	   resm <- list()
		for (m in 1:length(models)) { #m=1
		 #message(names(models)[m])
		 object <- models[[m]]  # str(models[[m]], max.level=1)
		 graph <- models[[m]]$graph
		 data <- models[[m]]$data
		 if (!is.null(outcome)) {
		  out <- model.matrix(~outcome-1)
		  colnames(out) <- gsub("outcome", "", colnames(out))
		  data <- cbind(out, data)
		  graph <- mapGraph(graph, type="outcome", C=ncol(out))
		  V(graph)$name[igraph::degree(graph, mode="out") == 0]<- colnames(out)
		 }

		 # Initialise K-fold cross-validation loop
		  pb0 <- pb(nrep = K, snow = FALSE)
		  resk <- list()
			for (k in 1:K) {
			  pb0$tick()
			  train <- c(1:nrow(data))[-idx[[k]]]
			  if (inherits(object, "SEM")) {
			   res0 <- quiet(SEMgraph::SEMrun(graph, data[train, ], algo="ricf", n_rep=0))
			  }
			  if (inherits(object, "ML")) {
			   ml0 <- c("SEM", "rpart", "ranger", "xgb.Booster")
			   ml1 <- c("sem", "tree", "rf", "xgb")
			  for (ml in 1:4){
			   if (inherits(object$model[[1]][[1]], ml0[ml])) algo<- ml1[ml]
			  }
			   res0<- quiet(SEMdeep::SEMml(graph, data[train, ], algo = algo))
			  }
			  if (inherits(object, "DNN")) { #str(object, max.level=1)
			   mp <- object$model[[1]][[1]]$param
			   mp <- object$model[[1]][[1]]$param
			   res0 <- quiet(SEMdeep::SEMdnn(graph, data[train, ], algo = mp[[1]],
				hidden = mp[[2]], link = mp[[3]], bias = mp[[4]], dropout = mp[[5]],
				loss = "mse", validation = mp[[7]], lambda = mp[[8]], alpha = mp[[9]],
				optimizer = "adam", lr = mp[[11]], batchsize = mp[[12]], shuffle = mp[[13]],
				baseloss = mp[[14]], burnin = mp[[15]], thr = NULL, nboot = 0,
				epochs = mp[[18]], patience = mp[[19]], device = "cpu", verbose = FALSE))
			  }
			  yhat0 <- predict(res0, data[-train, ])# dim(yhat0$Yhat)

			  if (is.null(outcome)) {
			   resk[[k]] <- data.frame(amse=yhat0$PE[1],r2=yhat0$PE[2],srmr=yhat0$PE[3])
			  }else{
			   yobs <- outcome[-train]
			   yhat <- yhat0$Yhat[ ,levels(outcome)]
			   resk[[k]] <- SEMdeep::classificationReport(yobs, yhat)$CM
			  }
			} # end K-folds loop

			if (is.null(outcome)) {
			 RESK <-  do.call(rbind, lapply(resk, as.data.frame))
			}else{
			 CMr <- base::Reduce('+', resk)
			 RESK <- SEMdeep::classificationReport(CM=CMr)$stats[C+2,c(3:5)]
			}
		  resm[[m]] <- apply(RESK, 2, mean)
		  message(" ", K, "-fold ", names(models)[m], " done.")
		} # end M-models loop

		RESM<- do.call(cbind, lapply(resm, matrix))
		rownames(RESM) <- names(resm[[1]])
		colnames(RESM) <- names(models)

	return(list(stats = t(RESM), PE = NULL))
}

createFolds <- function (y, k, list = TRUE, returnTrain = FALSE, ...) 
{
    # Here's a modified function from "caret" package @ Max Kuhn

    if (is.numeric(y)) {
        cuts <- floor(length(y)/k)
        if (cuts < 2) 
            cuts <- 2
        if (cuts > 5) 
            cuts <- 5
        breaks <- unique(quantile(y, probs = seq(0, 1, length = cuts)))
        y <- cut(y, breaks, include.lowest = TRUE)
    }
    if (k < length(y)) {
        y <- factor(as.character(y))
        numInClass <- table(y)
        foldVector <- vector(mode = "integer", length(y))
        for (i in 1:length(numInClass)) { #i=1
            min_reps <- numInClass[i]%/%k
            if (min_reps > 0) {
                spares <- numInClass[i]%%k
                seqVector <- rep(1:k, min_reps)
                if (spares > 0) 
                  seqVector <- c(seqVector, sample(1:k, spares))
                foldVector[which(y == names(numInClass)[i])] <- sample(seqVector)
            }
            else {
                foldVector[which(y == names(numInClass)[i])] <- sample(1:k, 
                  size = numInClass[i])
            }
        }
    }
    else foldVector <- seq(along = y)
    if (list) {
        out <- split(seq(along = y), foldVector)
        names(out) <- paste("Fold", gsub(" ", "0", format(seq(along = out))), 
            sep = "")
        if (returnTrain) 
            out <- lapply(out, function(data, y) y[-data], y = seq(along = y))
    }
    else out <- foldVector

    return(out)
}

pb <- function(nrep, snow = TRUE)
{
	pb <- progress::progress_bar$new(
			format = "(:spin) completed :current out of :total tasks [ :percent][ :elapsed]",
			total = nrep,      # Total number of ticks to complete.
			clear = TRUE,      # If TRUE, clears the bar when finish
			width = 80,        # Width of the progress bar (default=92)
			show_after = 0)    # Secs after which the bar is shown
	if (snow) {
	 progress <- function(x) pb$tick(tokens = list(trial = (1:nrep)[x]))
	}else{
	 progress <- pb
	}
	return(progress)
}

quiet <- function(..., messages=FALSE, cat=FALSE){
	if(!cat){
		tmpf <- tempfile()
		sink(tmpf)
		on.exit({sink(); file.remove(tmpf)})
	}
	out <- if(messages) eval(...) else suppressMessages(eval(...))
	out 
}
