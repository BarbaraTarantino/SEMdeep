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

#' @title Nodewise-predictive SEM train using Machine Learning (ML)
#'
#' @description The function converts a graph to a collection of 
#' nodewise-based models: each mediator or sink variable can be expressed as 
#' a function of its parents. Based on the assumed type of relationship, 
#' i.e. linear or non-linear, \code{SEMml()} fits a ML model to each
#' node (variable) with non-zero incoming connectivity. 
#' The model fitting is repeated equation-by equation (r=1,...,R) 
#' times, where R is the number of mediators and sink nodes. 
#'
#' @param graph An igraph object.
#' @param data A matrix with rows corresponding to subjects, and
#' columns to graph nodes (variables).
#' @param train A numeric vector specifying the row indices corresponding to
#' the train dataset (default = NULL). 
#' @param algo ML method used for nodewise-network predictions.
#' Six algorithms can be specified:
#' \itemize{
#' \item \code{algo="sem"} (default) for a linear SEM, see \code{\link[SEMgraph]{SEMrun}}. 
#' \item \code{algo="gam"} for a generalized additive model, see \code{\link[mgcv]{gam}}.
#' \item \code{algo="rf"} for a random forest model, see \code{\link[ranger]{ranger}}.
#' \item \code{algo="xgb"} for a XGBoost model, see \code{\link[xgboost]{xgboost}}.
#' \item \code{algo="nn"} for a small neural network model (1 hidden layer and 10 nodes), see \code{\link[nnet]{nnet}}.
#' \item \code{algo="dnn"} for a large neural network model (1 hidden layers and 1000 nodes), see \code{\link[cito]{dnn}}.
#' }
#' @param vimp A Logical value(default=FALSE). If TRUE compute the variable
#' importance, considering: (i) the squared value of the t-statistic or F-statistic
#' of the model parameters for "sem" or "gam"; (ii) the variable importance from
#' the \code{\link[ranger]{importance}} or \code{\link[xgboost]{xgb.importance}}
#' functions for "rf" or "xgb"; (iii) the Olden's connection weights for "nn" or
#' "dnn". 
#' @param thr A numerical value indicating the threshold to apply on the variable
#' importance to color the graph. If thr=NULL (default), the threshold is set to
#' thr = abs(mean(vimp)).
#' @param verbose A logical value. If FALSE (default), the processed graph
#' will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details By mapping data onto the input graph, \code{SEMml()} creates
#' a set of nodewise-based models based on the directed links, i.e., 
#' a set of edges pointing in the same direction, between two nodes 
#' in the input graph that are causally relevant to each other. 
#' The mediator or sink variables can be characterized in detail as 
#' functions of their parents. An ML model (sem, gam, rf, xgb, nn, dnn) 
#' can then be fitted to each variable with non-zero inbound connectivity, 
#' taking into account the kind of relationship (linear or non-linear). 
#' With R representing the number of mediators and sink nodes in the 
#' network, the model fitting process is performed equation-by-equation 
#' (r=1,...,R) times.
#'
#' @return An S3 object of class "ML" is returned. It is a list of 5 objects:
#' \enumerate{
#' \item "fit", a list of ML model objects, including: the estimated covariance 
#' matrix (Sigma),  the estimated model errors (Psi), the fitting indices (fitIdx),
#' and the signed Shapley R2 values (parameterEstimates), if shap = TRUE,
#' \item "Yhat", a matrix of predictions of sink and mediator graph nodes. 
#' \item "model", a list of all the fitted nodewise-based models 
#' (sem, gam, rf, xgb or nn).
#' \item "graph", the induced DAG of the input graph  mapped on data variables. 
#' If vimp = TRUE, the DAG is colored based on the variable importance measure,
#' i.e., if abs(vimp) > thr will be highlighted in red (vimp > 0) or blue
#' (vimp < 0). 
#' \item "data", input training data subset mapping graph nodes. 
#' }
#'
#' @author Mario Grassi \email{mario.grassi@unipv.it}
#'
#' @references 
#' 
#' Grassi M, Palluzzi F, Tarantino B (2022). SEMgraph: An R Package for Causal 
#' Network Analysis of High-Throughput Data with Structural Equation Models. 
#' Bioinformatics, 38 (20), 4829–4830 <https://doi.org/10.1093/bioinformatics/btac567>
#' 
#' Hastie, T. and Tibshirani, R. (1990) Generalized Additive Models. London: 
#' Chapman and Hall.
#' 
#' Breiman, L. (2001), Random Forests, Machine Learning 45(1), 5-32.
#' 
#' Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. 
#' Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge 
#' Discovery and Data Mining.
#' 
#' Ripley, B. D. (1996) Pattern Recognition and Neural Networks. Cambridge.
#' 
#' Redell, N. (2019). Shapley Decomposition of R-Squared in Machine Learning 
#' Models. arXiv: Methodology.
#'
#' @examples
#'
#' \donttest{
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
#' start<- Sys.time()
#' # ... rf
#' #res1<- SEMml(ig, data, train, algo="rf", vimp=FALSE)
#' res1<- SEMml(ig, data, train, algo="rf", vimp=TRUE)
#' 
#' # ... xgb
#' #res2<- SEMml(ig, data, train, algo="xgb", vimp=FALSE)
#' res2<- SEMml(ig, data, train, algo="xgb", vimp=TRUE)
#' 
#' # ... nn
#' #res3<- SEMml(ig, data, train, algo="nn", vimp=FALSE)
#' res3<- SEMml(ig, data, train, algo="nn", vimp=TRUE)
#' 
#' # ... gam
#' #res4<- SEMml(ig, data, train, algo="gam", vimp=FALSE)
#' res4<- SEMml(ig, data, train, algo="gam", vimp=TRUE)
#' end<- Sys.time()
#' print(end-start)
#'
#' # ... sem
#' #res5<- SEMml(ig, data, train, algo="sem", vimp=FALSE)
#' res5<- SEMml(ig, data, train, algo="sem", vimp=TRUE)
#'
#' #str(res5, max.level=2)
#' res5$fit$fitIdx
#' res5$fit$parameterEstimates
#' gplot(res5$graph)
#' 
#' #Comparison of AMSE (in train data)
#' rf <- res1$fit$fitIdx[2];rf
#' xgb<- res2$fit$fitIdx[2];xgb
#' nn <- res3$fit$fitIdx[2];nn
#' gam<- res4$fit$fitIdx[2];gam
#' sem<- res5$fit$fitIdx[2];sem
#' 
#' #Comparison of SRMR (in train data)
#' rf <- res1$fit$fitIdx[4];rf
#' xgb<- res2$fit$fitIdx[4];xgb
#' nn <- res3$fit$fitIdx[4];nn
#' gam<- res4$fit$fitIdx[4];gam
#' sem<- res5$fit$fitIdx[4];sem
#'
#' #Comparison of VIMP (in train data)
#' table(E(res1$graph)$color) #rf
#' table(E(res2$graph)$color) #xgb
#' table(E(res3$graph)$color) #nn
#' table(E(res4$graph)$color) #gam
#' table(E(res5$graph)$color) #sem
#'
#' #Comparison of AMSE (in test data)
#' print(predict(res1, data[-train, ])$PE[1]) #rf
#' print(predict(res2, data[-train, ])$PE[1]) #xgb
#' print(predict(res3, data[-train, ])$PE[1]) #nn
#' print(predict(res4, data[-train, ])$PE[1]) #gam
#' print(predict(res5, data[-train, ])$PE[1]) #sem
#' 
#' #...with a binary outcome (1=case, 0=control)
#' 
#' ig1<- mapGraph(ig, type="outcome"); gplot(ig1)
#' outcome<- ifelse(group == 0, -1, 1); table(outcome)
#' data1<- cbind(outcome, data); data1[1:5,1:5]
#' 
#' res6 <- SEMml(ig1, data1, train, algo="nn", vimp=TRUE)
#' gplot(res6$graph)
#' table(E(res6$graph)$color)
#' 
#' mse6 <- predict(res6, data1[-train, ])
#' yobs <- group[-train]
#' yhat <- mse6$Yhat[ ,"outcome"]
#' benchmark(yobs, yhat, thr=0, F1=TRUE)
#' benchmark(yobs, yhat, thr=0, F1=FALSE)
#' }
#' 
#' @export
#'

SEMml <- function(graph, data, train = NULL, algo = "sem", vimp = FALSE, thr = NULL, verbose = FALSE, ...) 
{
	# Set graph and data objects:
	nodes <- colnames(data)[colnames(data) %in% V(graph)$name]
	graph <- induced_subgraph(graph, vids=which(V(graph)$name %in% nodes))
	dag <- graph2dag(graph, data[train,], bap=FALSE) #del cycles & all <->
	din <- igraph::degree(dag, mode= "in")
	Vx <- V(dag)$name[din == 0]
	Vy <- V(dag)$name[din != 0]
	px <- length(Vx)
	py <- length(Vy)

	X <- data[, V(dag)$name]
	if (!is.null(train)) {
	  Z_train <- scale(X[train, ])
	  mp <- apply(Z_train, 2, mean)
	  sp <- apply(Z_train, 2, sd)
	  Z_test <- scale(X[-train, ], center=mp, scale=sp)
	}else{
	  Z_train <- Z_test <- scale(X)
	}
	n <- nrow(Z_train)

	# extract parameter estimates
	
	res <- parameterEstimates.ML(dag, Z_train, Z_test, algo, vimp)
	#str(res, max.level=1)

	if (vimp) {
	 class(res$est)<- c("lavaan.data.frame","data.frame")
	 df<- data.frame(res$est[,3],res$est[,1],weight=res$est[,4])
	 dag<- graph_from_data_frame(df)
	 if (is.null(thr)) thr <- mean(abs(E(dag)$weight), na.rm=TRUE)
	 dag<- colorDAG(dag, thr=thr, verbose=verbose)
	}
	
	# Shat and Sobs matrices :
	#Shat <- cor(cbind(Z_test[,Vx], res$yhat[,Vy]))
	#rownames(Shat) <- colnames(Shat) <- c(Vx,Vy)
	#Sobs <- cor(Z_test[, c(Vx,Vy)])
	Shat <- cor(cbind(Z_train[,Vx], res$YHAT[,Vy]))
	rownames(Shat) <- colnames(Shat) <- c(Vx,Vy)
	Sobs <- cor(Z_train[, c(Vx,Vy)])
	E <- Sobs - Shat # diag(E)

	# Fit indices : 
	SRMR <- sqrt(mean(E[lower.tri(E, diag = TRUE)]^2))
	if (algo == "sem") {
	 sem0 <- quiet(SEMrun(dag, Z_train, SE="none", limit=1000))
	 SRMR <- lavaan::fitMeasures(sem0$fit, "srmr")
	}
	logL <- -0.5 * (sum(log(res$sigma)) + py * log(n))#GOLEM, NeurIPS 2020
	AMSE <- mean(res$sigma, na.rm=TRUE) #average Mean Square Error
	idx <- c(c(logL=logL, amse=AMSE, rmse=sqrt(AMSE), srmr=SRMR))
	cat("\n", paste0(toupper(algo)," solver ended normally after ", py, " iterations"),"\n\n")
	cat(" logL:", idx[1], " srmr:", round(idx[4],7), "\n\n")

	fit <- list(Sigma=Shat, Beta=NULL, Psi=res[[2]], fitIdx=idx, parameterEstimates=res[[4]], formula=res[[5]])
	res <- list(fit=fit, Yhat=res[[3]], model=res[[1]], graph=dag, data=data[train,V(dag)$name])
	class(res) <- "ML"

	return(res)	
}

parameterEstimates.ML <- function(dag, Z_train, Z_test, algo, vimp, ...)
{
	# Set graph and data objects:
	V(dag)$name<- paste0("z", V(dag)$name)
	colnames(Z_train)<- paste0("z", colnames(Z_train))
	colnames(Z_test)<- paste0("z", colnames(Z_test))
	pe<- igraph::as_data_frame(dag)[,c(1,2)]
	y<- split(pe, f=pe$to)
	#y; length(y); names(y)
	sigma<- NULL
	#yhat<- NULL
	YHAT<- NULL
	est<- NULL
	ml<- list()
	fm<- list()

	for (j in 1:length(y)) {
	  cat(j, ":", names(y)[j], "\n")
	  Y <- names(y)[j]
	  X <- y[[j]][,1]
 	  C <- paste(X, collapse = " + ")
	  #f <- paste(Y,"~",X)
	  f <- formula(paste(Y,"~",C))
	  #pos <- 1
	  envir <- as.environment(1)
	  assign('f',f, envir = envir)

	  # fitting the model to predict Y on X
	  if (algo == "sem") {
		fit <- lm(eval(f), data = data.frame(Z_train))
		#fit <- glm(eval(f), family="binomial", data = data.frame(Z_train))
		if (vimp) vim <- (summary(fit)$coefficients[-1,3])^2
	  }
	  if (algo == "gam") {
		f <- formula(paste(Y,"~", paste("s(", X, ",bs='ps')", collapse="+")))
		fit <- mgcv::gam(eval(f), data = data.frame(Z_train))
	    if (vimp) vim <- as.numeric(summary(fit)$s.table[,3])
	  }
	  if (algo == "rf") {
		fit <- ranger::ranger(eval(f),
						data = Z_train,
						num.trees = 500, #default
						mtry = NULL,     #default
						importance = "impurity")
		if (vimp) vim <- as.numeric(ranger::importance(fit))
	  } 
	  if (algo == "xgb") {
		xgb_train <- xgboost::xgb.DMatrix(data = as.matrix(Z_train[, X]),
										  label = Z_train[, Y])
		fit <- xgboost::xgboost(data = xgb_train,
						booster = "gbtree",
						tree_method = "auto",
						max.depth = 6, #default
						nrounds = 100, #niter
						verbose = 0)   #silent																	
		fit$feature_names <- X
		vim <- 0
		if (vimp == TRUE & length(X) > 1) {
		 vimx<- xgboost::xgb.importance(model=fit)
		 vim <- unlist(vimx[,2])
		 names(vim) <- unlist(vimx[,1])
		 vim <- as.numeric(vim[X])
		}
	  }
	  if (algo == "nn") {
		fit <- nnet::nnet(eval(f),
						data = Z_train,
						linout = TRUE,#FALSE default
						size = 10,
						decay = 5e-3, #0 default
						maxit = 1000, #100 default
						trace = FALSE,#default
						MaxNWts = 1000) #default
		if (vimp) {
		 vimx <- NeuralNetTools::olden(fit)$data
		 vim <- vimx[,1]
		 names(vim) <- vimx[,2]
		 vim <- as.numeric(vim[X])
		}
	  }
	  if (algo == "dnn") {
		fit <- cito::dnn(eval(f),
						#as.formula(f),
						data = Z_train, 
						loss = "mse",
						hidden = 1000,
						activation = "selu",
						validation = 0,
						bias = TRUE,
						lambda = 0,
						alpha = 0.5,
						dropout = 0,
						optimizer = "adam",
						lr = 0.01,
						epochs = 32,
						plot = FALSE,
						verbose = FALSE,
						device = "cpu",
						early_stopping = FALSE)
		A <- matrix(rep(1, length(X)),ncol=1,
					dimnames = list(X,Y))#A
		if (vimp) vim <- as.numeric(getWeight(fit, A))
	  }

	if (vimp){
	 estj <- data.frame(
			 lhs = rep(gsub("z", "", Y), length(X)),
			 op = "~",
			 rhs = gsub("z", "", X),
			 varImp = vim)
	 est <- rbind(est, estj)
	}
	
	#TRAIN predictions and prediction error (MSE)
	if (algo == "rf"){
	  #pred <- predict(fit, Z_test)$predictions
	  PRED <- predict(fit, Z_train)$predictions
	  #if (OOB) pred <- rf.fit$predictions
	} else if (algo == "xgb"){
	  #pred <- predict(fit, data.matrix(Z_test[, fit$feature_names]))
	  PRED <- predict(fit, data.matrix(Z_train[, fit$feature_names]))
	} else{
	  #pred <- predict(fit, data.frame(Z_test))
	  PRED <- predict(fit, data.frame(Z_train))
	}
	 pe <- mean((Z_train[,Y] - PRED)^2)
	 sigma <- c(sigma, pe)
	 ml <- c(ml, list(fit))
	 fm <- c(fm, list(formula=f))
	 #yhat <- cbind(yhat, pred)
	 YHAT <- cbind(YHAT, PRED)
	}

	#colnames(yhat)<- sub(".", "", names(y))
	colnames(YHAT)<- sub(".", "", names(y))
	names(sigma) <- sub(".", "", names(y))
	
	return(list(ml = ml, sigma = sigma, YHAT = YHAT, est = est, formula = fm))
}

#' @title SEM-based out-of-sample prediction using node-wise ML
#'
#' @description Predict method for ML objects.
#'
#' @param object A model fitting object from \code{SEMml()} function. 
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
#' # Load Amyotrophic Lateral Sclerosis (ALS)
#' data<- alsData$exprs; dim(data)
#' data<- transformData(data)$data
#' ig<- alsData$graph; gplot(ig)
#' 
#' #...with train-test (0.5-0.5) samples
#' set.seed(123)
#' train<- sample(1:nrow(data), 0.5*nrow(data))
#'
#' start<- Sys.time()
#' # ... rf
#' res1<- SEMml(ig, data, train, algo="rf", vimp=FALSE)
#' mse1<- predict(res1, data[-train, ], verbose=TRUE)
#'
#' # ... xgb
#' res2<- SEMml(ig, data, train, algo="xgb", vimp=FALSE)
#' mse2<- predict(res2, data[-train, ], verbose=TRUE)
#' 
#' # ... nn
#' res3<- SEMml(ig, data, train, algo="nn", vimp=FALSE)
#' mse3<- predict(res3, data[-train, ], verbose=TRUE)
#' 
#' # ... gam
#' res4<- SEMml(ig, data, train, algo="gam", vimp=FALSE)
#' mse4<- predict(res4, data[-train, ], verbose=TRUE)
#'
#' # ... sem
#' res5<- SEMml(ig, data, train, algo="sem", vimp=FALSE)
#' mse5<- predict(res5, data[-train, ], verbose=TRUE)
#' end<- Sys.time()
#' print(end-start)
#' }
#'
#' @method predict ML
#' @export
#' @export predict.ML
#' 

predict.ML <- function(object, newdata, verbose=FALSE, ...)
{
	ml.fit <- object$model
	ml <- c("lm", "gam", "ranger", "xgb.Booster", "nnet", "citodnn")
	stopifnot(inherits(ml.fit[[1]], ml))
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))
	
	yhat <- NULL
	yn <- c()
	for (j in 1:length(ml.fit)) {
	 fit <- ml.fit[[j]]
	 vy <- all.vars(object$fit$formula[[j]])[1]
	 if (inherits(fit, "ranger")){
	  pred <- predict(fit, Z_test)$predictions
	 } else if (inherits(fit, "xgb.Booster")){
	  pred <- predict(fit, data.matrix(Z_test[,fit$feature_names]))
	 } else {
	  pred <- predict(fit, data.frame(Z_test))
	 }
	 yhat <- cbind(yhat, as.matrix(pred))
	 yn <- c(yn, vy)
	}

	yobs<- Z_test[, yn]
	PE<- colMeans((yobs - yhat)^2)
	pe<- mean((yobs - yhat)^2)
	colnames(yhat) <- gsub("z", "", yn)
	names(PE) <- colnames(yhat)
	if (verbose) print(c(amse=pe,PE))
	
	return(list(PE=c(amse=pe,PE), Yhat=yhat))
}

#' @title Compute variable importance using Shapley (R2) values
#'
#' @description This function computes variable contributions for individual
#' predictions using the Shapley values, a method from cooperative game
#' theory where the variable values of an observation work together to achieve
#' the prediction. In addition, to make variable contributions easily explainable, 
#' the function decomposes the entire model R-Squared (R2 or the coefficient
#' of determination) into variable-level attributions of the variance
#' (Redell, 2019).
#'
#' @param object A model fitting object from \code{SEMml()} function. 
#' @param newdata A matrix containing new data with rows corresponding to
#' subjects, and columns to variables.
#' @param thr A threshold value to apply to signed Shapley (R2) values. If
#' thr=NULL (default), the threshold is set to thr=mean(Shapley(R2) values)).
#' @param verbose A logical value. If FALSE (default), the processed
#' graph will not be plotted to screen.
#' @param ... Currently ignored.
#'
#' @details Lundberg & Lee (2017) proposed a unified approach to both
#' local explainability (the variable contribution of a single variable within
#' a single sample) and global explainability (the variable contribution of the
#' entire model) by applying the fair distribution of payoffs principles from
#' game theory put forth by Shapley (1953). Now called SHAP (SHapley Additive
#' exPlanations), this suggested framework explains predictions of ML models,
#' where input variables take the place of players, and their contribution to
#' a particular prediction is measured using Shapley values. 
#' Successively, Redell (2019) presented a metric that combines the additive 
#' property of Shapley values with the robustness of the R2 of Gelman (2018) 
#' to produce an R2 variance decomposition that accurately captures the 
#' contribution of each variable to the explanatory power of the model. 
#' Additionally, we use the signed R2, in order to denote the regulation
#' of connections in line with a linear SEM, since the edges in the DAG
#' indicate node regulation (activation, if positive; inhibition, if
#' negative). This has been recovered for each edge using sign(beta), i.e.,
#' the sign of the coefficient estimates from a linear model (lm) fitting
#' of the output node on the input nodes, as suggested by Joseph (2019).
#' It should be noted that in order to ascertain the local significance
#' of node regulation with respect to the DAG, the Shapley decomposition
#' of the R-squared (R2) value can be employed for each outcome node
#' (r=1,...,R) by averaging the R2 indices of their input nodes.
#'
#' The Shapley values are computed using the \pkg{shapr} package that implements
#' an extended version of the Kernel SHAP method for approximating Shapley values
#' in which dependence between the features is taken into account. 
#' The operations necessary to compute kernel SHAP values are inherently
#' time-consuming, with the computational time increasing in proportion to
#' the number of predictor variables and the number of observations.
#' Therefore, the function uses a progress bar to check the progress
#' of the kernel SHAP evaluation per observation.
#'
#' @return A list od three object: (i) data.frame including the connections
#' together with their signed Shapley R-squred values; (ii) the dag with
#' colored edges, if abs(sign_R2) > thr will be highlighted in red (sign_R2 > 0)
#' or blue (sign_R2 < 0); and (ii) the list of individual Shapley values of
#' predictors variables per each response variable.
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
#' Models. arXiv: Methodology.
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
#' rf0<- SEMml(ig, data, train=train, algo="rf", vimp=FALSE)
#' 
#' res<- getShapleyR2(rf0, data[-train, ], thr=NULL, verbose=TRUE)
#' table(E(res$dag)$color)
#'
#' # average shapley R2 across response variables
#' R2<- abs(res$est[,4])
#' Y<- res$est[,1]
#' R2Y<- aggregate(R2~Y,data=data.frame(R2,Y),FUN="mean")
#' PE<- predict(rf0, data[-train, ])$PE
#' cbind(R2Y=R2Y[,2],PEY=PE[-1])
#' mean(R2) # total average R2
#' PE[1]    # total MSE
#' }
#' 
#' @export
#'

getShapleyR2<- function(object, newdata, thr = NULL, verbose = FALSE, ...)
{
	# extract data and model objects
	model<- object$model
	graph<- object$graph
	formula<- object$fit$formula
	vp <- colnames(object$data)
	mp <- apply(object$data, 2, mean)
	sp <- apply(object$data, 2, sd)
	Z_train<- scale(object$data)
	Z_test <- scale(newdata[,vp], center=mp, scale=sp)
	colnames(Z_train)<- paste0("z", colnames(Z_train))
	colnames(Z_test) <- paste0("z", "", colnames(Z_test))
	est<- NULL
	shapx<- list()

	pb <- txtProgressBar(min = 0, max = length(model), style = 3)
	for (j in 1:length(model)) { #
	  fitj <- list(model[[j]],formula[[j]])
	  Wj <- shapR2(fitj, Z_train, Z_test)
	  vn <- all.vars(formula[[j]])
	  X <- gsub("z", "", vn[-1])
	  Y <- gsub("z", "", vn[1])
	  label<- data.frame(
				lhs = rep(Y, length(X)),
				op = "~",
				rhs = X)
	  est <- rbind(est, cbind(label, Wj[[2]]))
	  shapx<- c(shapx, list(Wj[[1]]))
	  setTxtProgressBar(pb, j)
	}
	cat("\n")
	rownames(est)<- NULL
	colnames(est)[4]<- "sign_r2"
	class(est)<- c("lavaan.data.frame","data.frame")

	df<- data.frame(est[,3],est[,1],weight=est[,4])
	dag<- graph_from_data_frame(df)
	if (is.null(thr)) thr = mean(abs(E(dag)$weight))
	dag<- colorDAG(dag, thr=thr, verbose=verbose)

	return(list(est = est, dag = dag, shapx = shapx))
}

shapR2<- function(fitj, Z_train, Z_test, ...)
{
	# Extract fitting 
	fit <- fitj[[1]]
	formula <- fitj[[2]]
	vf <- all.vars(formula)
	z_train <- data.frame(Z_train[ ,vf])
	z_test <- data.frame(Z_test[ ,vf])
		
	# Extract sign of beta coefficients from lm(Y on X)
	f0 <- paste(vf[1],"~", paste(vf[-1], collapse = " + "))
	fit0 <- lm(f0, data = data.frame(z_train))
	s <- sign(coefficients(fit0))[-1]
	p0 <- mean(z_train[,1])
	
	if (inherits(fit, "nnet")) {
	  predict_model.nnet.formula <- function(x, newdata) {
		predict(x, newdata, type = "raw")
	  }
	  #pos <- 1
	  envir <- as.environment(1)
	  assign("predict_model.nnet.formula", predict_model.nnet.formula, envir = envir)
 	}
	if (inherits(fit, "citodnn")) {
	  predict_model.citodnn <- function(x, newdata) {
		predict(x, newdata, type = "response")
	  }
	  #pos <- 1
	  envir <- as.environment(1)
	  assign("predict_model.citodnn", predict_model.citodnn, envir = envir)
 	}

	# Extract shapley data.table (data.frame)
	if (length(vf) != 2) {  
	 explainer <- quiet(shapr::shapr(z_train[,-1], fit,
						n_combinations = 1000))
	 explanation <- quiet(shapr::explain(z_test,
										explainer = explainer,
										approach = "empirical",
										prediction_zero = p0,
										n_combinations = 1000))
	 shapx <- data.frame(explanation$dt)[,-1]
	 #shapm <- s * apply(shapx, 2, function(x) mean(abs(x)))
	 shapr2 <- s * r2(shapx, z_test[,1], p0, scale="r2")[,3]
	} else {
	 if (inherits(fit, "ranger")){
	  shapx <- predict(fit, z_test)$predictions - p0
	 } else if (inherits(fit, "xgb.Booster")){
	  shapx <- predict(fit, data.matrix(z_test[, fit$feature_names])) - p0
	 } else {
	  shapx <- predict(fit, data.frame(z_test)) - p0
	 }
	 #shapm <- s * mean(abs(shapx))
	 shapr2 <- s * min(sum(shapx^2)/sum((z_test[,1] - p0)^2),1)
	}

	return(list(shapx = shapx, shapr2 = shapr2))
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
