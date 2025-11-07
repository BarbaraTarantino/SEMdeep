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

#' @title Nodewise SEM train using Machine Learning (ML)
#'
#' @description The function converts a graph to a collection of 
#' nodewise-based models: each mediator or sink variable can be expressed as 
#' a function of its parents. Based on the assumed type of relationship, 
#' i.e. linear or non-linear, \code{SEMml()} fits a ML model to each
#' node (variable) with non-zero incoming connectivity. 
#' The model fitting is performed equation-by equation (r=1,...,R) 
#' times, where R is the number of mediators and sink nodes. 
#'
#' @param graph An igraph object.
#' @param data A matrix with rows corresponding to subjects, and columns 
#' to graph nodes (variables).
#' @param outcome A character vector (as.fctor) of labels for a categorical
#' output (target). If NULL (default), the categorical output (target) will
#' not be considered.
#' @param algo ML method used for nodewise-based predictions.
#' Four algorithms can be specified:
#' \itemize{
#' \item \code{algo="sem"} (default) for a linear SEM, see \code{\link[SEMgraph]{SEMrun}}. 
#' \item \code{algo="tree"} for a CART model, see \code{\link[rpart]{rpart}}.
#' \item \code{algo="rf"} for a random forest model, see \code{\link[ranger]{ranger}}.
#' \item \code{algo="xgb"} for a XGBoost model, see \code{\link[xgboost]{xgboost}}.
#' }
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' variable importance values to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(variable importance values)).
#' @param nboot number of bootstrap samples that will be used to compute cheap
#' (lower, upper) CIs for all input variable weights. As a default, nboot = 0.
#' @param ncores number of cpu cores (default = 2)
#' @param verbose A logical value. If FALSE (default), the processed graph
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details By mapping data onto the input graph, \code{SEMml()} creates
#' a set of nodewise models based on the directed links, i.e., 
#' a set of edges pointing in the same direction, between two nodes 
#' in the input graph that are causally relevant to each other. 
#' The mediator or sink variables are defined as functions of their parents.
#' Then, an ML model (sem, tree, rf, xgb) can be fitted to each variable with
#' non-zero inbound connectivity. The model fitting process is performed
#' equation-by-equation (r=1,...,R) times, where R represents the number of
#' mediators and sink nodes in the input graph.
#'
#' If boot != 0, the function will implement the cheap bootstrapping proposed by
#' Lam (2002) to generate uncertainties (i.e., bootstrap \code{90\%CIs}) for ML
#' parameters. Bootstrapping can be enabled by setting a small number (1 to 10) of
#' bootstrap samples. Note, however, that the computation can be time-consuming for
#' massive MLs, even with cheap bootstrapping!
#'
#' @return An S3 object of class "ML" is returned. It is a list of 5 objects:
#' \enumerate{
#' \item "fit", a list of ML model objects, including: the estimated covariance 
#' matrix (Sigma), the estimated model errors (Psi), the fitting indices (fitIdx),
#' and the parameterEstimates, i.e., the variable importance measures (VarImp).
#' \item "gest", the data.frame of variable importances (parameterEstimates)
#' of outcome levels, if outcome != NULL.
#' \item "model", a list of all the fitted non-linear nodewise-based models 
#' (tree, rf, xgb, nn or dnn).
#' \item "graph", the induced DAG of the input graph  mapped on data variables. 
#' The DAG with colored edge/nodes based on the variable importance measures,
#' i.e., if abs(VarImp) > thr will be highlighted in red (VarImp > 0) or blue
#' (VarImp < 0). If the outcome vector is given, nodes with variable importances
#' summed over the outcome levels, i.e. sum(VarImp[outcome levels])) > thr,
#' will be highlighted in pink.
#' \item "data", input data subset mapping graph nodes.
#' }
#' Using the default \code{algo="sem"}, the usual output of a linear nodewise-based
#' SEM, see \code{\link[SEMgraph]{SEMrun}} (algo="cggm"), will be returned. 
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references 
#' 
#' Grassi M., Palluzzi F., and Tarantino B. (2022). SEMgraph: An R Package for Causal 
#' Network Analysis of High-Throughput Data with Structural Equation Models. 
#' Bioinformatics, 38 (20), 4829–4830 <https://doi.org/10.1093/bioinformatics/btac567>
#' 
#' Breiman L., Friedman J.H., Olshen R.A., and Stone, C.J. (1984) Classification
#' and Regression Trees. Chapman and Hall/CRC.
#' 
#' Breiman L. (2001). Random Forests, Machine Learning 45(1), 5-32.
#' 
#' Chen T., and Guestrin C. (2016). XGBoost: A Scalable Tree Boosting System. 
#' Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge 
#' Discovery and Data Mining.
#' 
#' Lam, H. (2022). Cheap bootstrap for input uncertainty quantification. WSC '22:
#' Proceedings of the Winter Simulation Conference, 2318-2329.
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
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' start<- Sys.time()
#' # ... tree
#' res1<- SEMml(ig, data[train, ], algo="tree")
#' 
#' # ... rf
#' res2<- SEMml(ig, data[train, ], algo="rf")
#' 
#' # ... xgb
#' res3<- SEMml(ig, data[train, ], algo="xgb")
#'
#' # ... sem
#' res4<- SEMml(ig, data[train, ], algo="sem")
#' 
#' end<- Sys.time()
#' print(end-start)
#'
#' #visualizaation of the colored dag for algo="sem"
#' gplot(res4$graph, l="dot", main="sem")
#'
#' #Comparison of fitting indices (in train data)
#' res1$fit$fitIdx #tree
#' res2$fit$fitIdx #rf
#' res3$fit$fitIdx #xgb
#' res4$fit$fitIdx #sem
#' 
#' #Comparison of parameter estimates (in train data)
#' parameterEstimates(res1$fit) #tree
#' parameterEstimates(res2$fit) #rf
#' parameterEstimates(res3$fit) #xgb
#' parameterEstimates(res4$fit) #sem
#' 
#' #Comparison of VarImp (in train data)
#' table(E(res1$graph)$color) #tree
#' table(E(res2$graph)$color) #rf
#' table(E(res3$graph)$color) #xgb
#' table(E(res4$graph)$color) #sem
#'
#' #Comparison of AMSE, R2, SRMR (in test data)
#' print(predict(res1, data[-train, ])$PE) #tree
#' print(predict(res2, data[-train, ])$PE) #rf
#' print(predict(res3, data[-train, ])$PE) #xgb
#' print(predict(res4, data[-train, ])$PE) #sem
#' 
#' #...with a categorical (as.factor) outcome
#' outcome <- factor(ifelse(group == 0, "control", "case")); table(outcome) 
#'
#' res5 <- SEMml(ig, data[train, ], outcome[train], algo="tree")
#' gplot(res5$graph)
#' table(E(res5$graph)$color)
#' table(V(res5$graph)$color)
#' 
#' pred <- predict(res5, data[-train, ], outcome[-train], verbose=TRUE)
#' yhat <- pred$Yhat[ ,levels(outcome)]; head(yhat)
#' yobs <- outcome[-train]; head(yobs)
#' classificationReport(yobs, yhat, verbose=TRUE)$stats
#' }
#' 
#' @export
#'
SEMml <- function(graph, data, outcome = NULL, algo = "sem", thr = NULL, nboot=0, ncores = 2, verbose = FALSE, ...) 
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
	
	# extract parameter estimates:
	res <- list()
	nboot <- nboot + 1
	pb0 <- pb(nrep = nboot, snow = FALSE)
	message("Running SEM model via ML...")
	
	  for (b in 1:nboot) {
	 	#set.seed(runif(1,0,1000))
		set.seed(b)
		if (b == 1) idx<- 1:nrow(data) else idx<- sample(nrow(data),replace = TRUE)
		if (algo == "sem") {
		 sem0 <- quiet(SEMrun(dag, data = data[idx, ], algo="cggm"))
		 res[[b]] <- list(ml=list(sem0), sigma=diag(sem0$fit[[3]])[Vy], YHAT=NULL)
		}else{
		 res[[b]] <- parameterEstimates.ML(dag, data = data[idx, ], algo, nboot, ncores)
		}
		pb0$tick()
		message(" done.")
	   #print(all.vars(res[[b]]$dnn[[1]]$old_formula));cat("\n")
	  }
	  # str(res[[1]], max.level=4)
	  #str(res, max.level=1)
	
	# Shat and Sobs matrices :
	if (algo == "sem") {
	 Shat<- res[[1]]$ml[[1]]$fit$Sigma[c(Vx,Vy),c(Vx,Vy)]
	}else{
	 Shat <- cor(cbind(data[,Vx], res[[1]]$YHAT[,Vy]))
	 rownames(Shat) <- colnames(Shat) <- c(Vx,Vy)
	}
	Sobs <- cor(data[, c(Vx,Vy)])
	E <- Sobs - Shat # diag(E)

	# Fit indices : 
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	Psi <- res[[1]]$sigma
	logL <- -0.5 * (sum(log(Psi)) + py * log(nrow(data)))#GOLEM, NeurIPS 2020
	AMSE <- mean(Psi, na.rm=TRUE) #average Mean Square Error
	idx <- c(c(logL=logL, amse=AMSE, rmse=sqrt(AMSE), srmr=SRMR))
	message("\n", toupper(algo), " solver ended normally after ", py, " iterations")
	message("\n", " logL:", round(idx[1],6), "  srmr:", round(idx[4],6))
	
	# Get variable Importance:
	model<- lapply(1:length(res), function(x) res[[x]][[1]])
	gest <- NULL
	#if (vimp) {
	 if (!is.null(outcome)) {
	  gest$lhs <- levels(outcome)
	  ml <- list(model=model, gest=gest, graph=dag-levels(outcome))
	 }else{
	  ml <- list(model=model, gest=gest, graph=dag)
	 }
	 vimp <- getVariableImportance(object=ml, thr=thr, verbose=FALSE)
	 est <- vimp$est
	 gest<- vimp$gest
	 dag <- vimp$dag
	#}

	fit <- list(Sigma=Shat, Beta=NULL, Psi=Psi, fitIdx=idx, parameterEstimates=est)
	res <- list(fit=fit, gest=gest, model=model, graph=dag, data=data)
	class(res) <- "ML"

	return(res)
}

parameterEstimates.ML <- function(dag, data, algo, nboot, ncores, ...)
{
	# Set data, graph and formula objects:
	Z_train<- scale(data)
	colnames(Z_train) <- paste0("z", colnames(Z_train))
	V(dag)$name <- paste0("z", V(dag)$name)
	pe<- igraph::as_data_frame(dag)[,c(1,2)]
	y <- split(pe, f=pe$to)
	#y; length(y); names(y)
	f <- list()
	vx <- list()
	n <- nrow(Z_train)

	for( j in 1:length(y)){
	 vx[[j]] <- y[[j]][,1]
	  if (length(vx[[j]]) >= n) {
	    set.seed(1)
		vx[[j]] <- vx[[j]][sample(1:length(vx[[j]]), round(0.9*n))]
	  }
 	 C <- paste(vx[[j]], collapse = " + ")
	 f[[j]] <- formula(paste(names(y)[j], "~", C))
	}

	#fitting a ML model to predict Y on X
	#pos <- 1
	envir <- as.environment(1)
	assign('f', f, envir = envir)
	nrep <- length(y)

	#opb<- pbapply::pboptions(type = "timer", style = 2)
	pb0 <- pb(nrep = nrep, snow = FALSE)

	if (algo == "tree") {
	 #fit <- pbapply::pblapply(1:nrep, function(x){
	 fit <- lapply(1:nrep, function(x){
	  pb0$tick()
	  cp<- rpart::rpart.control(cp = 0.01) #default cp
	  ml.fit <- rpart::rpart(eval(f[[x]]),
						data = data.frame(Z_train),
						model = TRUE, control = cp)
	 })
	 #}, cl=NULL)
	}

	if (algo == "rf") {
	 #fit <- pbapply::pblapply(1:nrep, function(x){
	 fit <- lapply(1:nrep, function(x){
	   pb0$tick()
	   ml.fit <- ranger::ranger(eval(f[[x]]),
						data = Z_train,
						num.trees = 500, #default
						mtry = NULL,     #default
						importance = "permutation", #"impurity"
						num.threads = ncores) #2
	 })
	 #}, cl=NULL)
	}

	if (algo == "xgb") {
	 #fit <- pbapply::pblapply(1:nrep, function(x){
	 fit <- lapply(1:nrep, function(x){
	  pb0$tick()
	  X <- vx[[x]]
	  Y <- names(y)[x]
	  ml.fit <- xgboost::xgboost(data = as.matrix(Z_train[, X]),
						label = Z_train[, Y],	
						booster = "gbtree", #tree based model
						tree_method = "hist", #faster hist algo
						max.depth = 6, #default
						nrounds = 100, #niter
						#device = "cpu", #default
						nthread = ncores, #-1, use all cores
						verbose = 0)   #silent			 
	 })
	 #}, cl=NULL)												
	}

	sigma<- NULL
	YHAT<- NULL
	for (j in 1:length(y)) {
	#TRAIN predictions and prediction error (MSE)
	 X <- vx[[j]]
	 Y <- names(y)[j]
	 if (algo == "rf"){
	   PRED <- predict(fit[[j]], Z_train, num.threads = ncores)$predictions
	 } else if (algo == "xgb"){
	   PRED <- predict(fit[[j]], as.matrix(Z_train[, X]))
	 } else{
	   PRED <- predict(fit[[j]], data.frame(Z_train))
	 }
	  pe <- mean((Z_train[ ,Y] - PRED)^2)
	  sigma <- c(sigma, pe)
	  YHAT <- cbind(YHAT, PRED)
	}

	colnames(YHAT) <- sub(".", "", names(y))
	names(sigma) <- sub(".", "", names(y))
	fit$old_formula <- f
	
	return(list(ml = fit, sigma = sigma, YHAT = YHAT))
}

#' @title SEM-based out-of-sample prediction using nodewise ML
#'
#' @description Predict method for ML objects.
#'
#' @param object A model fitting object from \code{SEMml()} function. 
#' @param newdata A matrix containing new data with rows corresponding to subjects,
#' and columns to variables.
#' @param newoutcome A new character vector (as.factor) of labels for a categorical
#' output (target)(default = NULL).
#' @param ncores number of cpu cores (default = 2)
#' @param verbose Print predicted out-of-sample MSE values (default = FALSE).
#' @param ... Currently ignored.
#'
#' @return A list of 3 objects:
#' \enumerate{
#' \item "PE", vector of the amse = average MSE over all (sink and mediators)
#' graph nodes; r2 = 1 - amse; and srmr= Standardized Root Means Squared Residual
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
#' # Load Amyotrophic Lateral Sclerosis (ALS)
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
#' # ... tree
#' res1<- SEMml(ig, data[train, ], algo="tree")
#' mse1<- predict(res1, data[-train, ], verbose=TRUE)
#'
#' # ... rf
#' res2<- SEMml(ig, data[train, ], algo="rf")
#' mse2<- predict(res2, data[-train, ], verbose=TRUE)
#' 
#' # ... xgb
#' res3<- SEMml(ig, data[train, ], algo="xgb")
#' mse3<- predict(res3, data[-train, ], verbose=TRUE)
#' 
#' # ... sem
#' res4<- SEMml(ig, data[train, ], algo="sem")
#' mse4<- predict(res4, data[-train, ], verbose=TRUE)
#' end<- Sys.time()
#' print(end-start)
#'
#' #...with a categorical (as.factor) outcome
#' outcome <- factor(ifelse(group == 0, "control", "case")); table(outcome) 
#'
#' res5 <- SEMml(ig, data[train, ], outcome[train], algo="tree")
#' pred <- predict(res5, data[-train, ], outcome[-train], verbose=TRUE)
#' yhat <- pred$Yhat[ ,levels(outcome)]; head(yhat)
#' yobs <- outcome[-train]; head(yobs)
#' classificationReport(yobs, yhat, verbose=TRUE)$stats
#' }
#'
#' @method predict ML
#' @export
#' @export predict.ML
#' 
predict.ML <- function(object, newdata, newoutcome=NULL, ncores=2, verbose=FALSE, ...)
{
	stopifnot(inherits(object, "ML"))
	if (inherits(object$model[[1]][[1]], "SEM")) {
 	 return(predict.SEM(object$model[[1]][[1]], newdata, newoutcome, verbose))
	}
	ml.fit <- object$model[[1]]
	fm <- ml.fit[length(ml.fit)]
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

	nrep <- length(ml.fit)-1
	yhat <- NULL
	yn <- c()
	
	for (j in 1:nrep) {
	 fit <- ml.fit[[j]]
	 vy <- all.vars(fm$old_formula[[j]])[1]
	 vx <- all.vars(fm$old_formula[[j]])[-1]
	 if (inherits(fit, "ranger")){
	  pred <- predict(fit, Z_test, num.threads = ncores)$predictions
	 } else if (inherits(fit, "xgb.Booster")){
	  pred <- predict(fit, as.matrix(Z_test[,vx]))
	 } else {
	  pred <- predict(fit, data.frame(Z_test))
	 }
	 yhat <- cbind(yhat, as.matrix(pred))
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

#' @title Variable importance for Machine Learning models
#'
#' @description Extraction of ML variable importance measures.
#'
#' @param object A model fitting object from \code{SEMml()} function.
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' variable importance values to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(variable importance values)).
#' @param verbose A logical value. If FALSE (default), the processed graph 
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details The variable (predictor) importance will be computed considering:
#' (i) the absolute value of the z-statistic of the model parameters for "sem";
#' and (ii) the variable importance measures from the \code{\link[rpart]{rpart}},
#' \code{\link[ranger]{importance}} or \code{\link[xgboost]{xgb.importance}} functions
#' for "tree", "rf" or "xgb" methods.
#'
#' @return A list of three object: (i) est: a data.frame including the connections together
#' with their variable importances (VarImp)), (ii) gest: if the outcome vector is given,
#' a data.frame of VarImp for outcome lavels, and (iii) dag: DAG with colored edges/nodes.
#' If abs(VarImp) > thr will be highlighted in red (VarImp > 0) or blue (VarImp < 0). If
#' the outcome vector is given, nodes with variable importances summed over the outcome
#' levels, i.e. sum(VarImp[outcome levels])) > thr, will be highlighted in pink.
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' add references
#'
#' @examples
#'
#' \donttest{
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#'
#' #ncores<- parallel::detectCores(logical = FALSE)
#' ml0<- SEMml(ig, data, outcome=NULL, algo="rf", ncores=2)
#'
#' vi05<- getVariableImportance(ml0, thr=0.5, verbose=TRUE)
#' table(E(vi05$dag)$color)
#' }
#'
#' @export
#' 
getVariableImportance <- function(object, thr = NULL, verbose = FALSE, ...)
{
	#stopifnot(inherits(object, c("SEM", "ML")))
	ml0 <- c("SEM", "rpart", "ranger", "xgb.Booster")
	stopifnot(inherits(object$model[[1]][[1]], ml0))
		
	# Set ML objects
	dag <- object$graph #gplot(dag)
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	  dag <- mapGraph(dag, type="outcome", C=length(out))
	  V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}
	
	nboot<- length(object$model)
	est_boot <- list()
	pb0 <- pb(nrep = nboot , snow = FALSE)
	
	for(b in 1:nboot) { #b=1
	  pb0$tick()
	   
	  fit <- object$model[[b]]
	  formula <- fit$old_formula
	  nrep <- length(fit) - 1
	  est <- NULL
	
		for (j in 1:nrep) {
		 # Extract Variable Importances for ML models
		 V <- all.vars(formula[[j]])
		 X <- V[-1]
		 Y <- V[1]
		
		 if (inherits(fit[[1]], "SEM")) {
			#n_e <- ecount(dag)
			est <- fit[[1]]$fit$parameterEstimates[ ,c(1:3,6)]
			break
		 }
		 if (inherits(fit[[1]], "rpart")) {
			vim <- fit[[j]]$variable.importance
			vim <- as.numeric(vim[X])
		 }
		 if (inherits(fit[[1]], "ranger")) {
			vim <- ranger::importance(fit[[j]])
			vim <- as.numeric(vim[X])
		 } 
		 if (inherits(fit[[1]], "xgb.Booster")) {
			fit[[j]]$feature_names <- X
			vim <- 1
			if (length(X) > 1) {
			vimx <- xgboost::xgb.importance(model=fit[[j]])
			vim <- unlist(vimx[,2])
			names(vim) <- unlist(vimx[,1])
			vim <- as.numeric(vim[X])
			}
		 }

		 if (length(vim) == 0) vim<- rep(NA, length(X))
		  estj <- data.frame(
				lhs = rep(sub(".", "", Y), length(X)),
				op = "~",
				rhs = sub(".", "", X),
				varImp = vim)
		  est <- rbind(est, estj)
		 }
	  
	  rownames(est) <- NULL
	  colnames(est)[4] <- "VarImp"
	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag)) 
}

#' @title Compute variable importance using Shapley (R2) values
#'
#' @description This function computes a model-agnostic variable importance
#' based on a Shapley-value decomposition of the model R-Squared (R2, i.e., the
#' coefficient of determination) that allocates the proportion of model-
#' explained variability in the data to each model feature (Redell, 2019).
#'
#' @param object A model fitting object from \code{SEMml()}, or \code{SEMrun()} 
#' functions.
#' @param newdata A matrix containing new data with rows corresponding to subjects,
#' and columns to variables.
#' @param newoutcome A new character vector (as.factor) of labels for a categorical
#' output (target)(default = NULL).
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' signed Shapley R2 to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(signed Shapley R2 values)).
#' @param ncores number of cpu cores (default = 2)
#' @param verbose A logical value. If FALSE (default), the processed
#' graph will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details Shapley values (Shapley, 1953; Lundberg & Lee, 2017) apply the fair
#' distribution of payoffs principles from game theory to measure the additive
#' contribution of individual predictors in a ML model. The function compute
#' a signed Shapley R2 metric, that combines the additive property of Shapley
#' values with the robustness of the R-squared (R2) of Gelman (2018) to produce
#' a variance decomposition that accurately captures the contribution
#' of each variable in the ML model, see Redell (2019). The signed values are
#' used in order to denote the regulation of connections in line with a linear
#' model, since the edges in the DAG indicate node regulation (activation, if
#' positive; inhibition, if negative). The sign has been recovered for each edge
#' using sign(beta), i.e., the sign of the coefficient estimates from a linear
#' model (lm) fitting of the output node on the input nodes (see Joseph, 2019).
#' Furthermore, to determine the local significance of node regulation in the DAG,
#' the Shapley decomposition of the R-squared values for each outcome node (r=1,...,R)
#' can be done by summing the Shapley R2 indices of their input nodes.
#' It should be noted that the operations required to compute Shapley values processed
#' with the \code{\link[kernelshap]{kernelshap}} function of the \pkg{kernelshap} R
#' package are inherently time-consuming, with the computational time increasing in
#' proportion to the number of predictor variables and the number of observations.
#'
#' @return A list od four object: (i) shapx: the list of individual Shapley values
#' of predictors variables per each response variable; (ii) est: a data.frame including
#' the connections together with their signed Shapley R-squred values; (iii) gest:
#' if the outcome vector is given, a data.frame of signed Shapley R-squred values per
#' outcome levels; and (iv) dag: DAG with colored edges/nodes. If abs(sign_r2) > thr
#' and sign_r2 < 0, the edge is inhibited and it is highlighted in blue; otherwise,
#' if abs(sign_r2) > thr and sign_r2 > 0, the edge is activated and it is highlighted
#' in red. If the outcome vector is given, nodes with absolute connection weights
#' summed over the outcome levels, i.e. sum(abs(sign_r2[outcome levels])) > thr, will
#' be highlighted in pink.
#' 
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#' 
#' Shapley, L. (1953) A Value for n-Person Games. In: Kuhn, H. and Tucker, A., 
#' Eds., Contributions to the Theory of Games II, Princeton University Press,
#' Princeton, 307-317. 
#' 
#' Scott M. Lundberg, Su-In Lee. (2017). A unified approach to interpreting 
#' model predictions. In Proceedings of the 31st International Conference on 
#' Neural Information Processing Systems (NIPS'17). Curran Associates Inc., 
#' Red Hook, NY, USA, 4768–4777.
#' 
#' Redell, N. (2019). Shapley Decomposition of R-Squared in Machine Learning 
#' Models. arXiv preprint: https://doi.org/10.48550/arXiv.1908.09718
#' 
#' Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). R-squared for 
#' Bayesian Regression Models. The American Statistician, 73(3), 307–309.
#'
#' Joseph, A. Parametric inference with universal function approximators (2019).
#' Bank of England working papers 784, Bank of England, revised 22 Jul 2020. 
#'
#' @examples
#'
#' \donttest{
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#'
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' rf0<- SEMml(ig, data[train, ], algo="rf")
#' 
#' res<- getShapleyR2(rf0, data[-train, ], thr=NULL, verbose=TRUE)
#' table(E(res$dag)$color)
#'
#' # shapley R2 per response variables
#' R2<- abs(res$est[,4])
#' Y<- res$est[,1]
#' R2Y<- aggregate(R2~Y,data=data.frame(R2,Y),FUN="sum");R2Y
#' r2<- mean(R2Y$R2);r2
#' }
#'
#' @export
#'
getShapleyR2<- function(object, newdata, newoutcome = NULL, thr = NULL, ncores = 2, verbose = FALSE, ...)
{
	stopifnot(inherits(object, c("SEM", "ML")))
	
	# extract model, formula and data objects
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}
	if (inherits(object, "SEM") | inherits(object$model[[1]][[1]], "SEM")){
	 pe <- igraph::as_data_frame(dag)[,c(1,2)]
	 pe$from <- paste0("z", pe$from)
	 pe$to <- paste0("z", "", pe$to)
	 y <- split(pe, f=pe$to)
	 model <- rep(0, length(y))
	 formula <- lapply(1:length(y), function(x){
					C<- paste(y[[x]][,1], collapse = " + ")
					f<- formula(paste(names(y)[x], "~", C))})
	} else {
	 model <- object$model[[1]]
	 formula <- model$old_formula
	}

	if (inherits(object, "SEM")) object$data<- object$data[ ,-1]
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	Z_train <- scale(object$data)
	if (!is.null(newoutcome)) {
	 newout <- model.matrix(~newoutcome-1)
	 colnames(newout) <- gsub("newoutcome", "", colnames(newout))
	 newdata <- cbind(newout, newdata)
	}
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_train) <- paste0("z", colnames(Z_train))
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))

	est <- NULL
	shapx <- list()
	nrep <- length(formula)

	#pb0 <- txtProgressBar(min = 0, max = length(model), style = 1)
	pb0 <- pb(nrep = nrep, snow = FALSE)

	for (j in 1:nrep) {
	 pb0$tick()
	 S <- shapR2(model[[j]], formula[[j]], Z_train, Z_test, ncores)
	 W <- S$shapr2
	 if (is.null(W)) {
	  message("\nWARNING: Model (", j,") skipped. Currently less samples than features not supported!")
	  next
	 }
	 vx <- rownames(W) <- sub(".", "", rownames(W))
	 vy <- colnames(W) <- sub(".", "", colnames(W))
	   for (k in 1:length(vy)) {
		label <- data.frame(
				 lhs = vy[k],
				 op = "~",
				 rhs = vx)
		est <- rbind(est, cbind(label, W[vx,vy[k]]))
	   }
	 shapx<- c(shapx, list(S$shapx))
	 #setTxtProgressBar(pb0, j)
	}
	#close(pb0)
	rownames(est) <- NULL
	colnames(est)[4] <- "sign_r2"

	estB <- est[!est[ ,1] %in% out, ]
	estB <- cbind(estB, lower=rep(0, nrow(estB)), upper=rep(0, nrow(estB)))
	gest <- est[est[ ,1] %in% out, ]
	gest <- cbind(gest, lower=rep(0, nrow(gest)), upper=rep(0, nrow(gest)))
	class(estB) <- c("lavaan.data.frame","data.frame")
	class(gest) <- c("lavaan.data.frame","data.frame")
	
	estC <- data.frame()
	if (length(out) > 0) {
	  C <- aggregate(abs(gest[ ,4])~gest$rhs,FUN="mean")
	  estC <- data.frame(
			 lhs = rep("outcome", nrow(C)),
			 op = "~",
			 rhs = C[ ,1],
			 sign_r2 = C[ ,2],
			 lower = rep(0, nrow(C)),
			 upper = rep(0, nrow(C)))
			 
	}else{
	 gest <- NULL
	}
	
	est <- list(estB = estB, estC = estC, gest = gest)
	dag0 <- colorDAG(dag, est, out, nboot=1, thr=thr, verbose=verbose)

	return(list(shapx = shapx, est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

shapR2<- function(fitj, fmj, Z_train, Z_test, ncores, ...)
{
	# Extract fitting objects
	vf <- all.vars(fmj)
	vx <- vf[-1]
	vy <- vf[1]
	z_train <- data.frame(Z_train[ ,vf])
	z_test <- data.frame(Z_test[ ,vf])
	if (nrow(z_train) < length(vx)) return(list(shapx = NULL, shapr2 = NULL))
 
	# Extract sign of beta coefficients from lm(Y on X)
	f0 <- paste("cbind(", paste(vy, collapse=","), ") ~ ", paste(vx, collapse="+"))
	fit0 <- lm(f0, data = z_train)
	if (length(vy) != 1) {
	 s <- apply(coef(fit0), c(1,2), sign)[-1, ]
	} else {
	 s <- sign(coef(fit0))[-1]
	}
	if (!is.list(fitj)) fitj<- fit0 #class(f0)

	# Extract predictions when number of predictors = 1
	if (length(vx) == 1) {
	  p0 <- rep(0, length(vy))
	  if (inherits(fitj, "ranger")){
	    pred <- predict(fitj, z_test, num.threads = ncores)$predictions
	  } else if (inherits(fitj, "xgb.Booster")){
	    pred <- predict(fitj, as.matrix(z_test[, vx]))
	  } else {
	    pred <- predict(fitj, data.frame(z_test))
	  }
	  R2 <- s * min(sum((pred - p0)^2)/sum((z_test[,vy] - p0)^2), 1)
	  if (length(vy) == 1) {
	   shapr2 <- as.matrix(R2)
 	   colnames(shapr2) <- vy
	  }else{
	   shapr2 <- t(as.matrix(R2))
	   rownames(shapr2) <- vx
	  }
	  return(list(shapx = NULL, shapr2 = shapr2))
	}

	# Calculate SHAP values with kernelshap
	if (inherits(fitj, "lm")) {
	  predict_model <- function(x, newdata) {
		predict(x, newdata)
	  }
 	}
	if (inherits(fitj, "rpart")) {
	  predict_model <- function(x, newdata) {
		predict(x, newdata, type = "vector")
	  }
 	}
	if (inherits(fitj, "ranger")) {
	  predict_model <- function(x, newdata) {
		predict(x, newdata, num.threads = ncores)$predictions
	  }
 	}
	if (inherits(fitj, "xgb.Booster")) {
	  predict_model <- function(x, newdata) {
		predict(x, as.matrix(newdata[ ,x$feature_names]))
	  }
 	}

    #pos <- 1
	envir <- as.environment(1)
	assign("predict_model", predict_model, envir = envir)

	ks <- kernelshap::kernelshap(fitj, z_test[ ,vx], exact = FALSE,
					   #parallel = NULL, parallel_args = NULL,
					   pred_fun = predict_model, verbose = FALSE)
	p0 <- ks$baseline

	if (length(vy) == 1) {
		ks$S <- list(ks$S)
		s <- as.matrix(s)
	}
	R2 <- lapply(1:length(vy), function(x)
			s[,x] * r2(ks$S[[x]], z_test[ ,vy[x]], p0[x], scale="r2")[,3])
	shapr2 <- do.call(cbind, lapply(R2, as.matrix))
	#R2T<- apply(shapr2, 2, function(x) sum(abs(x))); R2T
	colnames(shapr2) <- names(ks$S) <- vy

	return(list(shapx = ks$S, shapr2 = shapr2))
}

r2 <- function(shap, y, intercept, scale = c("r2", "1")) 
{
	shap <- as.data.frame(shap, drop = FALSE)
	scale <- scale[1]
	y <- as.vector(y)
	y_pred <- base::rowSums(shap, na.rm = TRUE) + intercept
	y_pred_var <- stats::var(y_pred)
	error_var <- stats::var(y - y_pred)
	r2 <- y_pred_var/(y_pred_var + error_var)
	
	data <- reshape(shap, varying = colnames(shap), timevar = "feature", 
					v.names = "shap_effect", direction = "long")[,-3]
	data$feature <- rep(colnames(shap), each=nrow(shap))
	rownames(data) <- NULL
	data$y <- as.vector(y)
	data$y_pred <- y_pred
	data$error <- data$y - data$y_pred
	y_pred_shap <- data$y_pred - data$shap_effect
	data$y_pred_shap <- y_pred_shap
	data$error_var <- error_var
	data$r2 <- r2
	data_r2_all <- c()
	data_sigma_all <- c()
  
	for (i in colnames(shap)){
	 data_r2 <- data[data$feature == i, ] 
	 data_r2$error_var_shap <- stats::var(data_r2$y - data_r2$y_pred_shap)
	 data_r2$error_ratio <- base::min(data_r2$error_var/data_r2$error_var_shap, 1, na.rm = TRUE)
	 data_r2_all <- rbind(data_r2_all, data_r2[1,])    
	
	 data_sigma <- data[data$feature == i, ]
	 data_sigma$error_var_shap <- stats::var(data_sigma$y - data_sigma$y_pred_shap)
	 data_sigma_all <- rbind(data_sigma_all, data_sigma[1,c(1,9)])
	}
	
	if ((base::sum(data_r2_all$r2 - data_r2_all$error_ratio * data_r2_all$r2)) == 0){
	 data_r2_all$r2_shap <- ((data_r2_all$r2 - data_r2_all$error_ratio * data_r2_all$r2)/1e-10)
	}else{
	 data_r2_all$r2_shap <- ((data_r2_all$r2 - data_r2_all$error_ratio * data_r2_all$r2)/
							(base::sum(data_r2_all$r2 - data_r2_all$error_ratio * data_r2_all$r2)))
	}
	
	error_var_shap <- sum(data_sigma_all$error_var_shap, na.rm = TRUE)
	var_ratio <- sum(error_var_shap - error_var, na.rm = TRUE)/
					(stats::var(y - intercept, na.rm = TRUE) - error_var)
	
	if (scale == "r2") {
	 data_r2_all$r2_shap <- data_r2_all$r2_shap * data_r2_all$r2
	}
	
	data_r2_all$sigma_unique <- var_ratio
	data <- as.data.frame(data_r2_all[, c("feature", "r2", "r2_shap", "sigma_unique")])
	
	return(data)
}

#' @title Compute variable importance using LOCO values
#'
#' @description This function computes the contributions of each variable
#' to individual predictions using LOCO (Leave Out COvariates) values.
#'
#' @param object A model fitting object from \code{SEMml()}, or \code{SEMrun()}
#' functions.
#' @param newdata A matrix containing new data with rows corresponding to subjects,
#' and columns to variables.
#' @param newoutcome A new character vector (as.factor) of labels for a categorical
#' output (target)(default = NULL).
#' @param thr A numeric value [0-1] indicating the threshold to apply to the
#' LOCO values to color the graph. If thr = NULL (default), the
#' threshold is set to thr = 0.5*max(abs(LOCO values)).
#' @param ncores number of cpu cores (default = 2)
#' @param verbose A logical value. If FALSE (default), the processed
#' graph will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details LOCO (Verdinelli & Wasserman, 2024) is a model-agnostic method for
#' assessing the importance of individual features (covariates) in a ML predictive
#' model. The procedure is simple: (i) train a model on the full dataset (with all
#' covariates) and (ii) for each covariate of interest: (a) remove (leave out) that
#' covariate from the dataset; (b) retrain the model on the remaining features; (c)
#' compare predictions between the full model and the reduced mode, and (d) evaluate
#' the difference in performance (e.g., using MSE, etc.).
#' LOCO is computationally expensive (requires retraining for each feature). The
#' getLOCO() function uses a lowest computation cost procedure (see Delicando & Pena,
#' 2023). The individual relevance of each variable is measured by comparing
#' the predictions of the model in the test set with those obtained when the variable
#' of interest is leave-out and substituted by its ghost variable in the test set.
#' This ghost variable is defined as the linear prediction of the covariate by
#' using the rest of the variables in the ML model. This method yields similar LOCO
#' results but requires much less computing time.
#'
#' @return A list od three object: (i) est: a data.frame including the connections
#' together with their LOCO values; (iii) gest: if the outcome vector is given,
#' a data.frame of LOCO values per outcome levels; and (iii) dag: DAG with colored
#' edges/nodes. If LOCO > thr, the edge is highlighted in red. If the outcome vector
#' is given, nodes with absolute connection weights summed over the outcome levels,
#' i.e. sum(LOCO[outcome levels]) > thr, will be highlighted in pink.
#' 
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references
#'
#' Verdinelli, I; Wasserman, L. Feature Importance: A Closer Look at Shapley Values
#' and LOCO. Statist. Sci. 39 (4) 623 - 636, November 2024. https://doi.org/10.1214/24-STS937
#'
#' Delicado, P.; Peña, D. Understanding complex predictive models with ghost variables.
#' TEST 32, 107–145 (2023). https://doi.org/10.1007/s11749-022-00826-x
#' 
#' @examples
#'
#' \donttest{
#' # load ALS data
#' ig<- alsData$graph
#' data<- alsData$exprs
#' data<- transformData(data)$data
#'
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' rf0<- SEMml(ig, data[train, ], algo="rf")
#' 
#' res<- getLOCO(rf0, data[-train, ], thr=0.2, verbose=TRUE)
#' table(E(res$dag)$color)
#' }
#'
#' @export
#'
getLOCO<- function(object, newdata, newoutcome = NULL, thr = NULL, ncores = 2, verbose = FALSE, ...)
{
	stopifnot(inherits(object, c("SEM", "ML")))
		
	# extract data, model, formula and data objects
	dag <- object$graph
	out <- unique(object$gest$lhs)
	if (length(out) > 0) {
	 dag <- mapGraph(dag, type="outcome", C=length(out))
	 V(dag)$name[igraph::degree(dag, mode="out") == 0] <- out
	}

	if (inherits(object, "SEM")){
	 object$data<- object$data[ ,-1]
	 model <- list(object)
	 fm <- NULL
	} else if (inherits(object$model[[1]][[1]], "SEM")){
	 model <- object$model
	 fm <- NULL
	} else {
	 model <- object$model 
	 fm <- model[[1]]$old_formula
	}

	if (!is.null(newoutcome)) {
	 newout <- model.matrix(~newoutcome-1)
	 colnames(newout) <- gsub("newoutcome", "", colnames(newout))
	 newdata <- cbind(newout, newdata)
	}
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))

	#str(model, max.level=2)
	nboot <- length(model)
	est_boot <- list()
	pb0 <- pb(nrep = nboot , snow = FALSE)

	for(b in 1:nboot) {
	  pb0$tick()
	  	  
	  est <- estLOCO(model[[b]], fm, Z_test, ncores)
	  
	  rownames(est) <- NULL
	  colnames(est)[4] <- "loco"
	  est_boot[[b]] <- est   
	}
	# str(est_boot, max.level=1)

	EST <- do.call(cbind, lapply(est_boot, data.frame))
	est <- cheapBootCI(EST[ ,c(1:3,seq(4, nboot*4, 4))], out, nboot)
	dag0 <- colorDAG(dag, est, out, nboot, thr, verbose=verbose)

	return(list(est = dag0$est, gest = dag0$gest, dag = dag0$dag))
}

estLOCO<- function(fit, fm, Z_test, ncores, ...)
{
	# extract LOCO for SEM
	if (inherits(fit, "SEM")) {
	 est <- parameterEstimates(fit$fit)
	 est <- est[est$op == "~", c(1:3,4)]
	 est[ ,4] <- est[ ,4]^2
	 return(est)
	}
	if (inherits(fit[[1]], "SEM")) {
	 est <- fit[[1]]$fit$parameterEstimates[ ,c(1:3,4)]
	 est[ ,4] <- est[ ,4]^2
	 return(est)
	}

	# start loop for ML
	nrep <- length(fm)
	est <- NULL

	for (j in 1:nrep) {
	  # extract fitting objects
	  fitj <- fit[[j]]
	  vf <- all.vars(fm[[j]])
	  vx <- vf[-1]
	  vy <- vf[1]
	  Z <- data.frame(Z_test[ ,vf])
	  X <- Z[ ,vx]
	  Y <- Z[ ,vy]

	  if (inherits(fitj, "ranger")){
	   Yhat <- predict(fitj, Z, num.threads = ncores)$predictions
	  } else if (inherits(fitj, "xgb.Booster")){
	   Yhat <- predict(fitj, as.matrix(Z[,vx]))
	  } else {
	   Yhat <- predict(fitj, Z)
	  }

		D2 <- matrix(0, nrow=length(vx), ncol=length(vy))
		for (k in 1:length(vx)) { #k=1
		 Zk <- Z #; print(head(Zk))
		 vk <- which(colnames(Zk) %in% vx[k])

		 if (length(vx) > 1){
		  C <- paste(vx[-k], collapse = " + ")
		  f <- formula(paste(vx[k], "~", C)) #; print(f)
		  Zk[ ,vk] <- predict(lm(eval(f), data = X)) #;print(head(Zk))
		 } else {
		  Zk[ ,vk] <- mean(Y) #;print(head(Zk))
		 }

		 if (inherits(fitj, "ranger")){
		  Yhatk <- predict(fitj, Zk, num.threads = ncores)$predictions
		 } else if (inherits(fitj, "xgb.Booster")){
		  Yhatk <- predict(fitj, as.matrix(Zk[,vx]))
		 } else {
		  Yhatk <- predict(fitj, Zk)
		 }

		 D2[k,] <- colMeans((as.matrix(Yhat - Yhatk))^2)
		} 
		rownames(D2) <- vx <- sub(".", "", vx)
		colnames(D2) <- vy <- sub(".", "", vy)
		#print(D2); cat("\n")
		 for (i in 1:length(vy)) {
			esti <- data.frame(
					  lhs = vy[i],
					  op = "~",
					  rhs = vx,
					  est = D2[,i])
			est <- rbind(est, esti)
		 }			  
	 #print(est); cat("\n")
	}

	return(est)
}
