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
#' The predictive procedure consists of two steps: (1) construction of the
#' topological layer (TL) ordering of the input graph; (2) prediction of
#' the node y values in a layer, where the nodes included in the previous
#' layers act as predictors x. 
#' 
#' @param object An object, as that created by the function \code{SEMrun()}
#' with the argument \code{fit} set to \code{fit = 0} or \code{fit = 1}.
#' @param newdata A matrix with new data, with rows corresponding to subjects,
#' and columns to variables. If \code{object$fit} is a model with the group
#' variable (\code{fit = 1}), the first column of newdata must be the new
#' group binary vector (0=control, 1=case). 
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details  The function first creates a layer-based structure of the
#' input graph. Then, a SEM-based predictive approach (Rooij et al., 2022) 
#' is used to produce predictions while accounting for the graph structure
#' organised in topological layers, j=1,...,L. In each iteration, the response
#' variables y are the nodes in the j layer and the predictors x are the nodes
#' belonging to the previous j-1 layers. 
#' Predictions (for y given x) are based on the (joint y and x) model-implied 
#' variance-covariance (Sigma) matrix and mean vector (Mu) of the fitted SEM,
#' and the standard expression for the conditional mean of a multivariate normal
#' distribution. Thus, the layer structure described in the SEM is taken into
#' consideration, which differs from ordinary least squares (OLS) regression.
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
#' @references 
#' 
#' de Rooij M, Karch JD, Fokkema M, Bakk Z, Pratiwi BC, and Kelderman H
#' (2023). SEM-Based Out-of-Sample Predictions, Structural Equation Modeling:
#' A Multidisciplinary Journal, 30:1, 132-148
#' <https://doi.org/10.1080/10705511.2022.2061494>
#'
#' Grassi M, Palluzzi F, Tarantino B (2022). SEMgraph: An R Package for Causal Network
#' Analysis of High-Throughput Data with Structural Equation Models.
#' Bioinformatics, 38 (20), 4829–4830 <https://doi.org/10.1093/bioinformatics/btac567>
#' 
#' @examples
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
#' # SEM fitting
#' #sem0<- SEMrun(ig, data[train,], algo="lavaan", SE="none")
#' #sem0<- SEMrun(ig, data[train,], algo="ricf", n_rep=0)
#' sem0<- SEMrun(ig, data[train,], SE="none", limit=1000)
#' 
#' # predictors, source+mediator; outcomes, mediator+sink
#'
#' res0<- predict(sem0, newdata=data[-train,]) 
#' print(res0$PE)
#' 
#' # SEM fitting
#' #sem1<- SEMrun(ig, data[train,], group[train], algo="lavaan", SE="none")
#' #sem1<- SEMrun(ig, data[train,], group[train], algo="ricf", n_rep=0)
#' sem1<- SEMrun(ig, data[train,], group[train], SE="none", limit=1000)
#'
#' # predictors, source+mediator+group; outcomes, source+mediator+sink
#'
#' res1<- predict(sem1, newdata=cbind(group,data)[-train,]) 
#' print(res1$PE)
#'
#' #...with a binary outcome (1=case, 0=control)
#' 
#' ig1<- mapGraph(ig, type="outcome"); gplot(ig1)
#' outcome<- ifelse(group == 0, -1, 1); table(outcome)
#' data1<- cbind(outcome, data); data1[1:5,1:5]
#' 
#' sem10 <- SEMrun(ig1, data1[train,], SE="none", limit=1000)
#' res10<- predict(sem10, newdata=data1[-train,], verbose=TRUE) 
#' 
#' yobs<- group[-train]
#' yhat<- res10$Yhat[,"outcome"]
#' performance(yobs, yhat)
#'
#' #...with predictors, source nodes; outcomes, sink nodes
#' ig2<- mapGraph(ig, type= "source"); gplot(ig2)
#' 
#' sem02 <- SEMrun(ig2, data[train,], SE="none", limit=1000)
#' res02<- predict(sem02, newdata=data[-train,], verbose=TRUE) 
#' #print(res02$PE)
#'
#' \donttest{
#' #...with 10-iterations of 10-fold cross-validation samples
#'
#' res<- NULL
#' for (r in 1:10) {
#'   set.seed(r)
#'   cat("rep = ", r, "\n")
#'   idx <- SEMdeep:::createFolds(y=data[,1], k=10)
#' 	for (k in 1:10) {
#' 	 cat("  k-fold = ", k, "\n")
#' 	 semr<- SEMdeep:::quiet(SEMrun(ig, data, SE="none", limit=1000))
#' 	 resr<- predict(semr, newdata=data[-idx[[k]], ])
#' 	 res<- rbind(res, resr$PE)
#' 	}
#' }
#' #res
#' apply(res, 2, mean)
#' }
#'
#' @method predict SEM
#' @export
#' @export predict.SEM
#' 

predict.SEM <- function(object, newdata, verbose = FALSE, ...)
{
	# set data, graph, predictors and outcomes
	stopifnot(inherits(object$fit, c("lavaan", "RICF", "GGM")))
	#stop("ERROR: in SEMrun( ) the fit=2 argument does not run (for now)")
	graph<- object$graph
	data <- object$data
	vnames <- colnames(data)
    if (!is.na(data[1, 1])) {
	 graph <- mapGraph(graph, type = "group")
	}else{
	 data <- data[,-1]
	 vnames	<- vnames[-1]
    }
	train <- data[,vnames]
	test <- newdata[,vnames]
	data <- rbind(train, test) #dim(data); head(data)
	vnodes <- colnames(data)[colnames(data) %in% V(graph)$name]
	graph<- induced_subgraph(graph, vids = vnodes)
	graph<- graph2dag(graph, data[train,])
	din<- igraph::degree(graph, mode = "in")
	dout<- igraph::degree(graph, mode = "out")
	V(graph)$color[din == 0]<- "cyan"
	V(graph)$color[dout == 0]<- "orange"
	L<- buildLevels(graph)
	if (verbose) gplot(graph)

	# SEM fitting on train data and predition on test data
	K_fold<- 1
	idx<- list(1:nrow(train))
	yobs<- NULL
	yhat<- NULL
	
	for (k in 1:K_fold) { #k=1
	  if (K_fold != 1) {
		message("Fold: ", k) 
		fit<- quiet(SEMrun(graph, data[idx[[k]],], algo="ricf", n_rep=0))
	  }else{
		fit<- object
	  }
	  if (inherits(fit$fit, "lavaan")) {
		#Sigma<- lavaan::fitted(fit$fit)$cov
		Sigma<- lavaan::lavInspect(fit$fit, "sigma")
		colnames(Sigma)<- gsub("z", "", colnames(Sigma))
		rownames(Sigma)<- colnames(Sigma)
		#mu<- fitted(fit$fit)$mean
	  }else{
		Sigma<- fit$fit$Sigma
		#mu<- rep(0, p)
	  }
	  Zk<- data[idx[[k]],]
	  mk<- apply(Zk, 2, mean)
	  sk<- apply(Zk, 2, sd)
	  yobsk<- NULL
	  yhatk<- NULL
		 
	  for (l in 1:(length(L)-1)) {
		yn<- L[[l]]
		xn<- unlist(L[(l+1):length(L)])
		Sxx<- Sigma[xn, xn]
		Sxy<- Sigma[xn, yn]
		mx<- rep(0, length(xn))
		my<- rep(0, length(yn))
		xtest<- as.matrix(data[-idx[[k]], xn])
		xtest<- scale(xtest, center=mk[xn], scale=sk[xn])
		#xtest<- scale(xtest, center = mx, scale = TRUE)
		n<- nrow(xtest)
		py<- length(yn)
		My<- matrix(my, n, py, byrow = TRUE)
		if (corpcor::is.positive.definite(Sxx)) {
			yhatlk<- My + xtest %*% solve(Sxx) %*% Sxy 
		}else{
			yhatlk<- My + xtest %*% Sxy
		}
		yobslk<- data[-idx[[k]], yn]
		yobslk<- scale(yobslk, center=mk[yn], scale=sk[yn])
		#yobslk<- scale(data[-idx[[k]], yn]) #dim(yhatlk)
		colnames(yobslk)<- colnames(yhatlk)<- yn
		yobsk<- cbind(yobsk, yobslk)
		yhatk<- cbind(yhatk, yhatlk)
	  }
	  
	  yobs<- rbind(yobs, yobsk)
	  yhat<- rbind(yhat, yhatk)
	}

	PE<- colMeans((yobs - yhat)^2)
	pe<- mean((yobs - yhat)^2)
	if (verbose) print(c(amse=pe,PE))

	return(list(PE=c(amse=pe,PE), Yhat=yhat))
}

createFolds <- function (y, k = 10, list = TRUE, returnTrain = TRUE, ...) 
{
    #createFolds() function from "caret" package (author: Max Kuhn)
	#All rights reserved. See the file COPYING for license terms.

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

#' @title Create a plot for a neural network model
#'
#' @description The function draws a neural network plot as a neural
#' interpretation diagram using with the \code{\link[NeuralNetTools]{plotnet}}
#' function of the \pkg{NeuralNetTools} R package.
#'
#' @param dnn.fit A neural network model from \pkg{cito} R package. 
#' @param bias A logical value, indicating whether to draw biases in 
#' the layers (default = FALSE). 
#' @param ... Currently ignored.
#'
#' @details The induced subgraph of the input graph mapped on data 
#' variables. Based on the estimated connection weights, if the connection
#' weight W > 0, the connection is activated and it is highlighted in red;  
#' if W < 0, the connection is inhibited and it is highlighted in blue.  
#' 
#' @return nplot returns invisibly the graphical object representing the
#' neural network architecture of NeuralNetTools. 
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
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#'
#' dnn0 <- SEMdnn(ig, data, train=1:nrow(data), grad = FALSE,
#' 			#loss = "mse", hidden = 5*K, link = "selu",
#' 			loss = "mse", hidden = c(10, 10, 10), link = "selu",
#' 			validation = 0, bias = TRUE, lr = 0.01,
#' 			epochs = 32, device = "cpu", verbose = TRUE)
#'
#'  for (j in 1:length(dnn0$model)) { 
#'    nplot(dnn0$model[[j]])
#'    Sys.sleep(5)
#'  }
#' }
#'
#' @export
#' 

nplot<- function(dnn.fit, bias=FALSE, ...)
{
	W<- coef(dnn.fit)[[1]] #str(W)
	w0<- list()
	for (i in seq(2, length(W), by=2)){
	  wi<- cbind(W[[i]],W[[i-1]])
	  w0<- c(w0, list(t(wi)))
	} #w0
	w1<- unlist(w0) #w1
	input<- dnn.fit$ model_properties$input
	hidden<- dnn.fit$ model_properties$hidden
	output<- dnn.fit$ model_properties$output
	struct1<- c(input, hidden, output)
	g<- NeuralNetTools::plotnet(w1, struct1, pos_col="red", neg_col="blue", bias=bias)
	
	return(invisible(g))
}

#' @title Map additional variables (nodes) to a graph object
#'
#' @description The function insert additional nodes to a graph object.
#' Among the node types, additional source or sink nodes can be added. 
#' Regarding the former, source nodes can represent: (i) data variables; 
#' (ii) a group variable; (iii) Latent Variables (LV). For the latter, an 
#' outcome variable, representing the prediction of interest, can be added. 
#' Moreover, \code{mapGraph()} can also create a new graph object starting
#' from a compact symbolic formula. 
#' 
#' @param graph An igraph object.
#' @param type A character value specifying the type of mapping. Five 
#' types can be specified. If \code{type = "source"} is specified, an 
#' additional source node (or more) is added to the graph. If 
#' \code{type = "group"}, an additional group source node is added. If
#' \code{type = "outcome"} (default), a prediction sink node is mapped
#' to the graph. If \code{type = "LV"}, a LV source node is included (where
#' the number of LV depends on the LV argument). If \code{type = "clusterLV"},
#' a series of clusters for the data are computed and a different LV source
#' node is added separately for each cluster.
#' @param LV The number of LV source nodes to add to the graph. This argument 
#' needs to be specified when \code{type = "LV"}. When \code{type = "clusterLV"}
#' the LV number is defined internally equal to the number of clusters.
#' (default = NULL).
#' @param f A formula object (default = NULL). A new graph object is created
#' according to the specified formula object. 
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
#' # ... map source nodes to ALS graph 
#' ig1 <- mapGraph(ig, type = "source"); gplot(ig1, l="dot")
#' 
#' # ... map group source node to ALS graph 
#' ig2 <- mapGraph(ig, type = "group"); gplot(ig2, l="fdp")
#' 
#' # ... map outcome sink to ALS graph 
#' ig3 <- mapGraph(ig, type = "outcome"); gplot(ig3, l="dot")
#' 
#' # ... map LV source nodes to ALS graph 
#' ig4 <- mapGraph(ig, type = "LV", LV = 3); gplot(ig4, l="fdp")
#' 
#' # ... map LV source nodes to the clusters of ALS graph 
#' ig5 <- mapGraph(ig, type = "clusterLV"); gplot(ig5, l="dot")
#'
#' # ... create a new graph with the formula variables
#' formula <- as.formula("z4747 ~ z1432 + z5603 + z5630")
#' ig6 <- mapGraph(f=formula); gplot(ig6)
#
#' @export
#' 

mapGraph <- function(graph, type = "outcome", LV = NULL, f = NULL, ...)
{
	if (!is.null(f)) {
	  g <- map_formula(f)
	  return(invisible(g))
	}
	if (type == "source") g <- map_source(graph)
	if (type == "group") g <- map_group(graph)
	if (type == "outcome") g <- map_outcome(graph)
	if (type == "LV") g <- map_LV(graph, LV=LV, cg=NULL)
	if (type == "clusterLV"){
	  cg<- SEMgraph::clusterGraph(graph, size=0)
	  LV<- length(table(cg))
	  g<- map_LV(graph, LV=LV, cg=cg)
	}
	return(invisible(g))
}

map_formula <- function(f, verbose=FALSE, ...)
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

map_source <- function(graph, verbose=FALSE, ...)
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

map_group <- function(graph, verbose=FALSE, ...)
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

map_outcome <- function(graph, verbose=FALSE, ...)
{
	gout <- graph + igraph::vertices("outcome")
	dout<- igraph::degree(graph, mode = "out")
	leaf<- V(graph)$name[dout == 0]
	E <- NULL
	 for(v in 1:length(leaf)){
	  	E <- c(E, leaf[v], "outcome")
	  }
	gout <- gout + igraph::edges(E)
	V(gout)$color[V(gout)$name == "outcome"] <- "green"
	if (verbose) gplot(gout)
	
	return(gout)	
}

map_LV <- function(graph, LV, cg=NULL, verbose=FALSE, ...)
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
	adj <- as_adj(dag, sparse=FALSE)
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

#' @title Prediction performance evaluation utility
#'
#' @description This function is able to calculate a series of binary 
#' classification evaluation statistics given (i) two vectors: one with the 
#' true target variable values, and the other with the predicted target variable
#' values or (ii) a confusion matrix with the counts for False Positives (FP), 
#' True Positives (TP), True Negatives (TN), and False Negatives (FN).
#' The user can specify the desired set of metrics to compute: (i) precision, 
#' recall, f1 score and Matthews Correlation Coefficient (mcc) or 
#' (ii) specificity, sensitivity, accuracy and mcc.
#' 
#' @param yobs A binary vector with the true target variable values. 
#' @param yhat A continuous vector with the predicted target variable values. 
#' @param CT An optional confusion matrix of dimension 2x2 containing the counts 
#' for FP, TP, TN, and FN.
#' @param thr A numerical value indicating the threshold for converting the
#' \code{yhat} continuous vector to a binary vector. If \code{yhat} vector 
#' ranges between -1 and 1, the user can specify \code{thr = 0} (default); 
#' if \code{yhat} ranges between 0 and 1, the user can specify \code{thr = 0.5}.
#' @param F1 A logical value. If TRUE (default), precision (pre), recall (rec),
#' f1 and mcc will be computed. Otherwise, if FALSE, specificity (sp),
#' sensitivity (se), accuracy (acc) and mcc will be obtained.
#' @param verbose A logical value. If FALSE (default), the density plots of 
#' \code{yhat} per group will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details #' Suppose a 2x2 table with notation
#'
#' \tabular{rcc}{ \tab Reference \tab \cr Predicted \tab Event \tab No Event
#' \cr Event \tab A \tab B \cr No Event \tab C \tab D \cr }
#'
#' The formulas used here are: \deqn{se = A/(A+C)} \deqn{sp =
#' D/(B+D)} \deqn{acc = (A+D)/(A+B+C+D)} \deqn{pre = A/(A+B)} 
#' \deqn{rec = A/(A+C)} \deqn{F1 = (2*pre*rec)/(pre+rec)} 
#' \deqn{mcc = (A*D - B*C)/sqrt((A+B)*(A+C)*(D+B)*(D+C))}
#'
#' @return A data.frame with classification evaluation statistics is returned.  
#'
#' @export
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references 
#' 
#' Sammut, C. & Webb, G. I. (eds.) (2017). Encyclopedia of Machine Learning 
#' and Data Mining. New York: Springer. ISBN: 978-1-4899-7685-7 
#' 
#' Chicco, D., Jurman, G. (2020) The advantages of the Matthews correlation 
#' coefficient (MCC) over F1 score and accuracy in binary classification 
#' evaluation. BMC Genomics 21, 6. 
#' 
#' @examples
#'
#' # Load Amyotrophic Lateral Sclerosis (ALS)
#' data<- alsData$exprs; dim(data)
#' data<- transformData(data)$data
#' group<- alsData$group; table (group)
#' ig<- alsData$graph; gplot(ig)
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#' 
#' #...with a binary outcome (1=case, 0=control)
#' ig1<- mapGraph(ig, type = "outcome"); gplot(ig1)
#' outcome<- group; table(outcome)
#' data1<- cbind(outcome, data); data1[1:5,1:5]
#' 
#' res <- SEMml(ig1, data1, train, algo="rf")
#' mse <- predict(res, data1[-train, ])
#' yobs<- group[-train]
#' yhat<- mse$Yhat[ ,"outcome"]
#' #yprob<- exp(yhat)/(1+exp(yhat))
#' 
#' # ... evaluate predictive performance (sp, se, acc, mcc)
#' performance(yobs, yhat, thr=0, F1=FALSE)
#' 
#' # ... evaluate predictive performance (pre, rec, f1, mcc)
#' performance(yobs, yhat, thr=0, F1=TRUE)
#' 
#' #... with confusion matrix table as input
#' ypred<- ifelse(yhat < 0, 0, 1)
#' performance(CT=table(yobs, ypred), F1=TRUE)
#' 
#' #...with density plots of yhat per group
#' old.par <- par(no.readonly = TRUE)
#' performance(yobs, yhat, thr=0, F1=FALSE, verbose = TRUE)
#' par(old.par)
#'
#' @export
#'

performance<- function(yobs, yhat, CT=NULL, thr=0, F1=TRUE, verbose=FALSE, ...)
{  
	# Confusion 2x2 table
	if (inherits(CT, "table")){
	 CT<- CT
	}else{
	 yobs<- factor(yobs, levels=c(0,1))
	if (thr == 0){
		thr<- (mean(yhat[yobs==0]) + mean(yhat[yobs==1]))/2
		ypred<- ifelse(yhat < thr, 0, 1)
		ypred<- factor(ypred, levels=c(0,1))
	} else if (thr > 0){
		wp1<- table(yobs)[2] / length(yobs)
		yhat<- classadjust(yhat, wrongprob1=wp1, trueprob1=wp1)
		ypred<- ifelse(yhat < 0.5, 0, 1)
		ypred<- factor(ypred, levels=c(0,1))
	}
	 CT<- table(yobs,ypred)
	}
	
	print(CT)
	cat("\n")
	a<- as.numeric(CT[2,2]) #TP
	b<- as.numeric(CT[2,1]) #FN
	c<- as.numeric(CT[1,2]) #FP
	d<- as.numeric(CT[1,1]) #TN
	
	sp<- d/(c+d) # Specificity, TN/(TN+FP)
	se<- a/(a+b) # Sensitivity, TP/(TP+FN)
	pre<- a/(a+c) # Precision, TP/(TP+FP)
	rec<- a/(a+b) # Recall, TP/(TP+FN)
	acc<- (a+d)/(a+b+c+d) # Accuracy, (TP+TN)/n
	f1<- (2*rec*pre)/(rec+pre) # F1=harmonic accuracy
	mcc<- (a*d - b*c)/sqrt((a+c)*(a+b)*(d+c)*(d+b)) # MCC
	
	if (verbose) dplot(yobs=yobs, yhat=yhat, thr=thr, fp=(1-sp), fn=(1-se))
	if (F1 == TRUE) return(data.frame(pre, rec, f1, mcc))
	if (F1 == FALSE) return(data.frame(sp, se, acc, mcc))
}

dplot<- function(yobs, yhat, thr, fp, fn, ...)
{ 
	# density plot of yhat per yobs(0,1)
	xlim <- c(min(yhat)-0.5, max(yhat)+0.5)
	err <- c(paste0("FP = ",round(fp,3)),
			 paste0("FN = ",round(fn,3)))
	par(mfrow=c(2,1), mar=rep(3,4))
	for (c in 0:1) { 
		d <- density(yhat[yobs == c])
		x <- d$x
		y <- d$y/max(d$y)
		main <- paste0("group ", c, " (", err[c+1], ")")
		plot(x, y, type="l", xlim=xlim, main=main)
		polygon(x, y, col="gray", border="gray")
		if (c == 0) {
		 region.x <- x[thr <= x & x <= xlim]
		 region.y <- y[thr <= x & x <= xlim]
		}else{
		 region.x <- x[xlim <= x & x <= thr]
		 region.y <- y[xlim <= x & x <= thr]
		}
		region.x <- c(region.x[1], region.x, tail(region.x,1))
		region.y <- c(0, region.y, 0)
		polygon(region.x, region.y, density=-1, col="red") 
	}
}

classadjust <- function(condprobs, wrongprob1, trueprob1)
{
	wrongratio <- (1 - wrongprob1)/wrongprob1
	fratios <- (1/condprobs - 1) * (1/wrongratio)
	trueratios <- (1 - trueprob1)/trueprob1
	return(1/(1 + trueratios * fratios))
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
