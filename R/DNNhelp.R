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

#Helper functions for DNN models:

check_device <- function(device)
{
  if (device == "cuda"){
    if (torch::cuda_is_available()) {
      device <- torch::torch_device("cuda")}
    else{
      warning("No Cuda device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  } else if (device == "mps") {
     if (torch::backends_mps_is_available()) {
      device <- torch::torch_device("mps")}
     else{
      warning("No mps device detected, device is set to cpu")
      device <- torch::torch_device("cpu")
    }
  } else {
    if (device != "cpu") warning(paste0("device ",device," not know, device is set to cpu"))
    device <- torch::torch_device("cpu")
  }
  return(device)
}

get_data_loader <- function(X, Y, batchsize = NULL, shuffle = TRUE)
{
	# Convert R data to torch tensor
	X <- torch::torch_tensor(as.matrix(X), dtype = torch::torch_float32())
	Y <- torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float32())
	
	# Define the dataloader
	ds <- torch::tensor_dataset(X = X, Y = Y)
	if (is.null(batchsize)) batchsize <- round(0.1*nrow(X))
	dl <- torch::dataloader(ds, batchsize, shuffle)

	return(dl)
}

get_activation_layer <- function(activation)
{
  return(switch(tolower(activation),
                "relu" = torch::nn_relu(),
                "leaky_relu" = torch::nn_leaky_relu(),
                "tanh" = torch::nn_tanh(),
                "elu" = torch::nn_elu(),
                "rrelu" = torch::nn_rrelu(),
                "prelu" = torch::nn_prelu(),
                "softplus" = torch::nn_softplus(),
                "celu" = torch::nn_celu(),
                "selu" = torch::nn_selu(),
                "gelu" = torch::nn_gelu(),
                "relu6" = torch:: nn_relu6(),
                "sigmoid" = torch::nn_sigmoid(),
                "softsign" = torch::nn_softsign(),
                "hardtanh" = torch::nn_hardtanh(),
                "tanhshrink" = torch::nn_tanhshrink(),
                "softshrink" = torch::nn_softshrink(),
                "hardshrink" = torch::nn_hardshrink(),
                "log_sigmoid" = torch::nn_log_sigmoid(),
                stop(paste0(activation, " as an activation function is not supported"))
  ))
}

get_optimizer <- function (optimizer, parameters, lr)
{
  optimizer <- match.arg(tolower(optimizer), choices = c("sgd", 
            "adam", "adadelta", "adagrad", "rmsprop", "rprop"))

  optim <- switch(optimizer,
            adam = torch::optim_adam(params = parameters, lr = lr),
			
			adadelta = torch::optim_adadelta(params = parameters, lr = lr),
			
			adagrad = torch::optim_adagrad(params = parameters, lr = lr),
			
			rmsprop = torch::optim_rmsprop(params = parameters, lr = lr),
			
			rprop = torch::optim_rprop(params = parameters, lr = lr),
			
			sgd = torch::optim_sgd(params = parameters, lr = lr),
			
			stop(paste0("optimizer = ", optimizer, 
             " is not supported, choose between adam, adadelta, adagrad, rmsprop, rprop or sgd"))
		   )
 
  return(optim)
}

get_loss_function <- function(loss, ...) {
	return(switch(tolower(trimws(loss)),
	
		# Regression losses
		"mse" = torch::nn_mse_loss(...),
		"l1" = torch::nn_l1_loss(...),
		"smooth_l1" = torch::nn_smooth_l1_loss(...),
		"nll" = torch::nn_nll_loss(...),
		
		# Classification losses
		"bce" = torch::nn_bce_loss(...),
		"bce_logits" = torch::nn_bce_with_logits_loss(...),
		"cross_entropy" = torch::nn_cross_entropy_loss(...),
		"poisson" = torch::nn_poisson_nll_loss(...),
		
		# Advanced/specialized losses
		"kld" = torch::nn_kl_div_loss(...),
		"hinge" = torch::nn_hinge_embedding_loss(...),
		"cosine" = torch::nn_cosine_embedding_loss(...),
    
    stop(paste0("'", loss, "' is not a supported loss function!"))
  ))
}

regularize_weights <- function (parameters, lambda, alpha)
{
  weight_layers <- names(which(sapply(parameters, function(x) length(dim(x))) > 1))

  regularization <- torch::torch_zeros(1L, dtype = parameters[[1]]$dtype, device = parameters[[1]]$device)

  for (i in 1:length(weight_layers)) {
      l1 <- torch::torch_sum(torch::torch_abs(parameters[[weight_layers[i]]]))
      l1 <- l1$mul(1-alpha)
      l2 <- torch::torch_norm(parameters[[weight_layers[i]]], p=2L)
      l2 <- l2$mul(alpha)

      regularization_tmp <- torch::torch_add(l1,l2)
      regularization_tmp <- regularization_tmp$mul(lambda)
      regularization <- regularization$add(regularization_tmp)
  }

  return(regularization)
}

MaskedLinear <- torch::nn_module(
  classname = "MaskedLinear",
  
  initialize = function(input_dim, output_dim, mask = NULL) {
    # Initialize weights
    self$weight <- torch::nn_parameter(torch::torch_randn(output_dim, input_dim))
    self$bias <- torch::nn_parameter(torch::torch_zeros(output_dim))

    # Initialize mask (if none provided, create unit mask)
    if (is.null(mask)) {
      mask <- torch::torch_ones_like(self$weight)
    } else {
	  mask <- torch::torch_tensor(as.matrix(t(mask)), dtype = torch::torch_uint8())
	}
	# Register buffer for mask (non-trainable parameter)
    self$register_buffer("mask", mask)
    
    # Initialize weights with mask already applied
    torch::with_no_grad({
      self$weight$mul_(self$mask)
    })
  },
  
  forward = function(x) {
    # Apply mask each forward pass (in case weights were updated)
    masked_weight <- self$weight * self$mask
    x <- torch::nnf_linear(x, masked_weight, self$bias)
    return(x)
  },
  
  # Custom gradient handling
  apply_gradient_mask = function() {
    if (!is.null(self$weight$grad)) {
      # Apply mask to gradients
      self$weight$grad$mul_(self$mask)
    }
  }
)

GreedyFactorizer <- function(adjacency, opt_args = NULL) {
  obj <- list(
    adjacency = adjacency,
    opt_args = opt_args
  )
  
  # obj=NULL; adjacency=A; hidden_sizes=c(20,20)
  
  obj$factorize <- function(hidden_sizes) {
    masks <- list()
    adj_mtx <- adjacency
    
    for (layer in hidden_sizes) {
      factorized <- obj$factorize_single_mask_greedy(adj_mtx, layer)
      M1 <- factorized$M1
      M2 <- factorized$M2
      
      adj_mtx <- M1
      masks <- append(masks, list(t(M2)))
    }
    
    masks <- append(masks, list(t(M1)))
    return(masks)
  }
    
  obj$factorize_single_mask_greedy <- function(adj_mtx, n_hidden) { 
    A_nonzero <- adj_mtx[rowSums(adj_mtx != 0) > 0, , drop = FALSE]
    n_nonzero_rows <- nrow(A_nonzero)
    
    M2 <- matrix(0, nrow = n_hidden, ncol = ncol(adj_mtx))
    for (i in 1:n_hidden) {
      M2[i, ] <- A_nonzero[(i - 1) %% n_nonzero_rows + 1, ]
    }
    
    M1 <- matrix(1, nrow = nrow(adj_mtx), ncol = n_hidden)
    for (i in 1:nrow(M1)) { #i=1
      Ai_zero <- which(adj_mtx[i, ] == 0)
      row_idx <- unique(which(M2[, Ai_zero] == 1, arr.ind = TRUE))#[, 1])
      M1[i, row_idx] <- 0
    }
    
    return(list(M1 = M1, M2 = M2))
  }
  
  return(obj)
}

ZukoFactorizer <- function(adjacency, opt_args = NULL) {
  obj <- list(
    adjacency = adjacency,
    opt_args = opt_args
  )
  
  # obj=NULL; adjacency=A; hidden_sizes=c(20,20)
  
    obj$factorize = function(hidden_sizes) {
      masks <- list()
           
      # Find unique rows and their inverse indices
      A_prime <- unique(adjacency)
      inv <- match(
        apply(adjacency, 1, paste, collapse = ","),
        apply(A_prime, 1, paste, collapse = ",")
      )
      
      # Compute n_deps and P matrix
      n_deps <- rowSums(A_prime)
      P <- (A_prime %*% t(A_prime) == n_deps) * 1.0
      
      indices <- NULL
	  n_outputs <- nrow(adjacency)
      all_hidden <- c(hidden_sizes, n_outputs)
      
      for (i in seq_along(all_hidden)) { #i=3
        h_i <- all_hidden[i]
        
        if (i > 1) {
          # Not the first mask
          mask <- P[, indices, drop = FALSE]
        } else {
          # First mask: just use rows from A
          mask <- A_prime
        }
        
        if (sum(mask) == 0) {
          stop("Adjacency matrix will yield null Jacobian.")
        }
        
        if (i <= length(hidden_sizes)) {
          # Still on intermediate masks
          reachable <- which(rowSums(mask) > 0)
          indices <- reachable[((1:h_i) - 1) %% length(reachable) + 1]
          mask <- mask[indices, , drop = FALSE]
        } else {
          # We are at the last mask
          mask <- mask[inv, , drop = FALSE]
        }
        
        # Need to transpose the mask to match dimensions
        masks[[i]] <- t(mask) #str(masks)
      }
      
      return(masks)
    }
	
  return(obj)
}

check_masks <- function(masks, A) {
  mask_prod <- masks[[1]]
  for (i in 2:length(masks)) {
    mask_prod <- mask_prod %*% masks[[i]]
  }
  mask_prod <- t(mask_prod)
  
  constraint <- (mask_prod > 0.0001) * 1 - A
  
  return(!any(constraint != 0))
}

DNN <- function(input, output, hidden, activation, bias, dropout)
{
	layers <- list()
	
	# Simple single layer network
	if (is.null(hidden)) {
	 layers[[1]] <- torch::nn_linear(input, out_features = output, bias = bias)
	} else {
	# Multi-layer network
	
	 # Replicate parameters to match hidden layers length
	 if (length(hidden) != length(activation)) activation <- rep(activation, length(hidden))
	 if (length(hidden)+1 != length(bias)) bias <- rep(bias, (length(hidden)+1))
	 if (length(hidden) != length(dropout)) dropout <- rep(dropout,length(hidden))

     # Build hidden layers
	 counter <- 1
	 for(i in 1:length(hidden)) {
	  # Add linear layer
	  if (counter == 1) {
		layers[[1]] <- torch::nn_linear(input, out_features = hidden[1], bias = bias[1])
	  } else {
		layers[[counter]] <- torch::nn_linear(hidden[i-1], out_features = hidden[i], bias = bias[i])
	  }
	  counter <- counter + 1
	  
	  # Add activation layer	
	  layers[[counter]] <- get_activation_layer(activation[i])
	  counter <- counter + 1
		
	  # Add dropout layer if specified
	  if (dropout[i] > 0) {
		layers[[counter]] <- torch::nn_dropout(dropout[i])
		counter <- counter + 1
	  }
	 }
	 
	 # Add output layer
	 layers[[length(layers)+1]] <- torch::nn_linear(hidden[i], out_features = output, bias = bias[i+1])
	}
	
	net <- do.call(torch::nn_sequential, layers)
	
	return(net)
}

StrNN <- torch::nn_module(
  classname = "StrNN",
   
  # Model initialization
  initialize = function(A, input, output, hidden, activation, bias, dropout, type, mask) {
   
    # set activation function
	self$activation <- get_activation_layer(activation)
	self$type <- type
	self$mask <- mask
  
	# Load adjacency matrix 
    if (!is.null(A)) {
      self$A <- A
    } else {
	   if (self$type == "made") {
        warning("Adjacency matrix is unspecified, defaulting to fully autoregressive structure.")
        self$A <- lower.tri(matrix(1, output, input))
       } else {
        stop("Adjacency matrix must be specified if factorizer is not MADE.")
       }
	}

	# Set masks
	 if (!is.null(self$mask)) {
      # Load precomputed masks if provided
      masks <- self$mask
    } else {
	  if (self$type == "greedy") {
		factorizer <- GreedyFactorizer(self$A)
	  #}
	  #else if (self$type == "made") {
	  #  factorizer <- MADEFactorizer(self$A)
	  }
	  else if (self$type == "zuko") {
		factorizer <- ZukoFactorizer(self$A)
	  }
      masks <- factorizer$factorize(hidden)
    }

	# Check masks
    if (!check_masks(masks, self$A)) {
       #stop("Mask check failed! Run a new DNN a single width hidden layer")
       warning("Mask check failed! Run a new DNN with a single width hidden layer")
    }

	# Define StrNN network
    self$layers <- torch::nn_module_list()
    hs <- c(input, hidden, output)
    
    # Build network with masked linear layers and activations
    for (i in 1:(length(hs) - 1)) {
      h0 <- hs[i]
      h1 <- hs[i + 1]
	  mask <- masks[[i]]
      
      # Add MaskedLinear layer
      self$layers$append(MaskedLinear(h0, h1, mask))
      
      # Add activation (except for the last layer)
      if (i < length(hs) - 1) {
        self$layers$append(self$activation)
      }
    }
	
    # Weight initialization
    self$apply(function(m) {
      if (inherits(m, "MaskedLinear")) {
        torch::nn_init_xavier_uniform_(m$weight)
        if (!is.null(m$bias)) torch::nn_init_normal_(m$bias, std = 0.01)
      }
    })
  },
  
  forward = function(x) {
    # Forward pass through the network
    for (i in 1:length(self$layers)) {
      x <- self$layers[[i]](x)
    }
    return(x)
  }
)

GAE <- torch::nn_module(
  classname = "GAE",
  
	# Model initialization
	initialize = function(A, input, output, hidden, activation, bias, dropout, type, mask) {    
      
	  # Create encoder and decoder
      self$encoder <- DNN(input, output, hidden, activation, bias, dropout)
      self$decoder <- DNN(input, output, hidden, activation, bias, dropout)
        
      # Initialize W with random uniform values
      self$d <- nrow(A)
	  self$W <- torch::nn_parameter(torch::torch_tensor(A, dtype = torch::torch_float(), requires_grad = TRUE))
      self$W <- torch::nn_init_uniform_(torch::torch_empty(self$d, self$d), -0.1, 0.1)
      
	  # Set mask matrix and GAE type
	  I <- torch::torch_eye(self$d)
	  if (is.null(mask)) {
	   self$mask <- 1 - I
      }
	  self$type <- type
	  
      # Weight initialization
      for (module in private$modules) {
       if (inherits(module, "nn_linear")) {
        torch::nn_init_xavier_normal_(module$weight)
        if (!is.null(module$bias)) {
          torch::nn_init_constant_(module$bias, 0)
        }
       }
      }
	},

	forward = function(x) { 
	  # Get adjacency matrix object
	  W <- self$W * self$mask
	  #print(round(as.matrix(self$W), 3))
	  
	  type <- self$type
	  
      if (type == "GAE0"){ 
		# Encoder
		h1 <- self$encoder$forward(x)
		
		# Compute z using adjacency matrix
		z <- torch::torch_matmul(h1, W)
		
		# Decoder
		x_hat <- self$decoder$forward(z)
	  }
	  
	  if (type == "GAE1"){ 
		B <- I - self$W
		C <- torch::torch_inverse(B) - I
		
		# Encoder
		h1 <- self$encoder$forward(x)
		
		# Compute Z = h1*(I - W)
		z <- torch::torch_matmul(h1, B)
		
		# Compute X = z*inv(I - W)
		#y <- torch::torch_matmul(z, C)
				
		# Decoder
		#x_hat <- self$decoder$forward(y)
		h2 <- self$decoder$forward(z)
		
		# Compute X = h2*inv(I - W)
		x_hat <- torch::torch_matmul(h2, C)
	  }

	  return(list(x_hat = x_hat, z_hat = z, W_hat = W))
	},

	# Acyclicity constraint using -log(det)
	h_func = function() {
      I <- torch::torch_eye(self$d)
      A <- self$W * (1 - I)
      #M <- I + (A * A) / self$d
      #
      ## Matrix power for approximating matrix exponential
      #E <- torch::torch_matrix_power(M, self$d)
      #h <- torch::torch_trace(E) - self$d
      #h <- torch::torch_trace(torch::torch_matrix_exp(A * A)) - self$d
      h <- -log(torch::torch_det(I - A * A)) #with s=1

	  return(h)
	} 
)

NGM <- torch::nn_module(
  classname = "NGM",

	# Model initialization
	initialize = function(A, input, output, hidden, activation, bias, dropout) {    

	  # Create MLP
      self$mlp <- DNN(input, output, hidden, activation, bias, dropout)
	  self$bias <- bias
	  
      # Set complement of A
	  self$Ac <- (A == 0) * 1
     	  
      # Weight initialization
      for (module in self$modules) {
       if (inherits(module, "nn_linear")) {
        torch::nn_init_xavier_normal_(module$weight)
        if (!is.null(module$bias)) {
          torch::nn_init_constant_(module$bias, 0)
        }
       }
      }
	},

	forward = function(x) { #x=X_train_batch
	  h <- self$mlp(x)

	  # Compute the product weight matrix: W = abs(W1)abs(W2) ... abs(WL)
	  param <- self$mlp$parameters #str(param)
	  W <- torch::torch_abs(param[[1]]$t())
	  W <- torch::nnf_normalize(W, dim = 2)
	  if (self$bias) K=seq(3,length(param),by=2) else K=2:length(param)
	  for (k in K) {
	   Wk <- torch::torch_abs(param[[k]]$t())
       Wk <- torch::nnf_normalize(Wk, dim = 2)
       W <- torch::torch_matmul(W, Wk)
	  }

      # Compute Hadamard product with Ac
      Wa <- W * self$Ac
 
	  return(list(x_hat = h, Wa = Wa))
	}
)

build_model <- function(algo, input, output, hidden, activation, bias, dropout, A=NULL, type=NULL, mask=NULL)
{
	if (algo == "nodewise" | algo == "layerwise") {
		net <- DNN(input = input, output = output,
				hidden = hidden, activation = activation,
				bias = bias, dropout = dropout)
		#net; str(net)
	}
	if (algo == "structured") {
		if (hidden[1] < input) hidden <- hidden + input #n_hid > n_inp 
		net <- StrNN(A = t(A), input = input, output = output,
				hidden = hidden, activation = activation,
				bias = bias, dropout = 0, type = "greedy", mask = NULL)
		#net; str(net)
	}
	if (algo == "neuralgraph") {
		net <- NGM(A = A, input = input, output = output,
				hidden = hidden, activation = activation,
				bias = bias, dropout = dropout)
		#net; str(net)
	}
	#if (algo == "autoencoder") {
	#	#opt <- list(l_1 = 0.01, l_2 = 0.01, l_3 = 10, h_tol = 1e-8)
	#	net <- GAE(A = A, input = input, output = output,
	#			hidden = hidden, activation = activation,
	#			bias = bias, dropout = dropout, type = "GAE0", mask = NULL)
	#	#net; str(net)
	#}
	return(net)
}

train_model <- function(data, algo, hidden, activation, bias, dropout,
					  loss, validation, lambda, alpha,
					  optimizer, lr, batchsize, shuffle, baseloss, burnin,
					  epochs, patience, device, verbose,
					  X, Y, A) {

	# Set device
	device <- check_device(device)

	# Set dataloader
	if (validation != 0) {
	 n_samples <- nrow(X)
	 valid <- sort(sample(c(1:n_samples), replace=FALSE, size = round(validation*n_samples)))
	 train <- c(1:n_samples)[-valid]
	 train_dl <- get_data_loader(X[train, ], Y[train, ], batchsize, shuffle)
	 valid_dl <- get_data_loader(X[valid, ], Y[valid, ], batchsize, shuffle)
	} else {
	 train_dl <- get_data_loader(X, Y, batchsize, shuffle)
	 valid_dl <- NULL
	}

	# Build model
	net <- build_model(algo = algo, input = ncol(X), output = ncol(Y),
						hidden = hidden, activation = activation,
						bias = bias, dropout = dropout, A = A)
	#net; str(net)

	# Define optimizer algorithm
	#optimizer <- torch::optim_adam(net$parameters, lr)
	optimizer <- get_optimizer(optimizer, net$parameters, lr)

	# Define loss functions
	#loss.fn <- torch::nnf_mse_loss
	loss.fn <- get_loss_function(loss)

	# Define base (NULL) loss
	if (is.null(baseloss)) {
	 y_base <- matrix(apply(Y, 2, mean), nrow(Y), ncol(Y), byrow = TRUE)
	 Y_base <- torch::torch_tensor(y_base, dtype = torch::torch_float32())
	 Y_torch <- torch::torch_tensor(as.matrix(Y), dtype = torch::torch_float32())
	 baseloss <- as.numeric(loss.fn(Y_base, Y_torch))
	}

	# Set object for training
	net$to(device = device)
	net$train()
	#regularize <- !(lambda == 0)
	best_train_loss <- Inf
	best_val_loss <- Inf
	losses <- data.frame(epoch=c(1:epochs),train_l=NA,valid_l= NA,train_l1= NA)
	train_weights <- list()
	train_state_dict <- list()
	counter <- 0

	for (epoch in 1:epochs) { #epoch=1
	  #Training batch loop
	  if (algo == "nodewise")
	   tbl <- batch_loop_DNN(train_dl, net, optimizer, loss.fn, lambda, alpha, device) # str(tbl)
	  if (algo == "layerwise") 
	   tbl <- batch_loop_DNN(train_dl, net, optimizer, loss.fn, lambda, alpha, device) # str(tbl)
	  if (algo == "structured")
	   tbl <- batch_loop_StrNN(train_dl, net, optimizer, loss.fn, device) # str(tbl)
	  if (algo == "neuralgraph")
	   tbl <- batch_loop_NGM(train_dl, net, optimizer, loss.fn, device) # str(tbl)
	  #if (algo == "autoencoder")
	  # tbl <- batch_loop_GAE(train_dl, net, optimizer, loss.fn, device) # str(tbl)
	  if(is.na(tbl$train_l[1])) {
		if(verbose) cat("Loss is NA. Bad training: change DNN learning parameters!\n")
		break
	  }

	  # average train loss
	  losses$train_l[epoch] <- mean(tbl$train_l)
	  losses$train_l1[epoch] <- mean(tbl$train_l1)

	  if (epoch >= burnin) {
		if (losses$train_l[epoch] > baseloss) {
		  if (verbose) cat("Stop training: loss is still above baseline!\n")
		  break
		}
	  }

	  # Loop for validation dl data
	  if (validation != 0 & !is.null(valid_dl)){
	  	net$train(FALSE)
	  	valid_l <- c()
	  
	  	coro::loop(for (batch in valid_dl) {
	  	    output <- net(batch$X$to(device = device, non_blocking = TRUE))
			if (algo != "neuralgraph") {
			  loss <- loss.fn(output, batch$Y$to(device = device, non_blocking = TRUE))
	        } else  {
			  loss <- loss.fn(output$x_hat, batch$Y$to(device = device, non_blocking = TRUE))
	        } 
	  	    valid_l <- c(valid_l, loss$item())
	  	})
	  
	  	losses$valid_l[epoch] <- mean(valid_l)
	  
	  	net$train(TRUE)
	  }

	  # Print progress
	  if (algo == "structured" | algo == "neuralgraph") {
	   if (validation != 0 & !is.null(valid_dl)) {
	  	 if (verbose & epoch %% 10 == 0) {
	  	    cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f, l1: %3.5f\n",
	  				    epoch, losses$train_l[epoch], losses$valid_l[epoch], losses$train_l1[epoch]))
	     }
	   } else {
	  	 if (verbose & epoch %% 10 == 0) {
	  		cat(sprintf("Loss at epoch %d: %3f, l1: %3.5f\n",
	  					epoch, losses$train_l[epoch], losses$train_l1[epoch]))
	     }
	   }
	  }

	  # Save best weights, state_dict and training loss
	  if (losses$train_l[epoch] < best_train_loss) {
		best_train_loss <- losses$train_l[epoch]
		train_weights[[1]] <- lapply(net$parameters, function(x) torch::as_array(x$to(device="cpu")))
		train_state_dict[[1]] <- lapply(net$state_dict(), function(x) torch::as_array(x$to(device="cpu")))
		counter <- 0
	  }

	  # Early stopping using validation loss
	  if (validation != 0) {
		if (losses$valid_l[epoch] < best_val_loss) {
		  best_val_loss <- losses$valid_l[epoch]
		  counter <- 0
	    }
	  }
	  if (counter >= patience) {
	    break
	  }
	  counter <- counter + 1
	}

	# Save last weights and state_dict
	train_weights[[2]] <- lapply(net$parameters, function(x) torch::as_array(x$to(device="cpu")))
	train_state_dict[[2]] <- lapply(net$state_dict(), function(x) torch::as_array(x$to(device="cpu")))

	# Set output list
	out <- list()
	out$net <- net
	out$loss <- c(train = best_train_loss, val = best_val_loss, base = baseloss)
	out$losses <- losses
	out$weights <- list(best_weights=train_weights[[1]], last_weights=train_weights[[2]])
	out$state_dict <- list(best_state_dict=train_state_dict[[1]], last_state_dict=train_state_dict[[2]])
	out$data <- list(X = X, Y = Y, A = A)
	if (validation != 0) out$data <- append(out$data, list(validation = valid))
	out$param <- list(algo = algo, hidden = hidden, link = activation, bias = bias,
				  dropout = dropout, loss = "mse", validation = validation,
				  lambda = lambda, alpha = alpha, optimizer = "adam", lr = lr,
				  batchsize = batchsize, shuffle = shuffle, baseloss = baseloss,
				  burnin = burnin, thr = NULL, nboot = 0, epochs = epochs,
				  patience = patience, device = "cpu", verbose = FALSE)

	class(out)<- "DNN"

	return(out) #str(out)
}

batch_loop_DNN <- function(train_dl, net, optimizer, loss.fn, lambda, alpha, device, ...)
{
	train_l <- c()
	train_l1 <- c()

	  # Training batch loop
	  coro::loop(for (batch in train_dl) {
		
		#batch=NULL;x=torch::torch_tensor(X[1:32,]);y=torch::torch_tensor(Y[1:32,]);batch$X=x;batch$Y=y
	
		# Zero gradients
		optimizer$zero_grad()

		# Forward pass
		output <- net(batch$X$to(device = device, non_blocking = TRUE))

		# Reconstruction loss
		#loss <- (output - batch$Y)$pow(2)$sum()/(nrow(batch$Y)*ncol(batch$Y))
		#loss <- (output - batch$Y)$pow(2)$mean()
		loss <- loss.fn(output, batch$Y$to(device = device, non_blocking = TRUE))
	
		if (lambda != 0) {
		 l12_loss <- regularize_weights(net$parameters, lambda, alpha)
		} else {
		 l12_loss <- 0
		}
		total_loss <- loss + l12_loss

		# Backward pass 
		total_loss$backward()
		
		#Update weights
		optimizer$step()

		train_l <- c(train_l, as.numeric(loss))
		train_l1 <- c(train_l1, as.numeric(l12_loss))
	  })

	return(list(output = output, train_l = train_l, train_l1 = train_l1))
}

batch_loop_StrNN <- function(train_dl, net, optimizer, loss.fn, device, ...)
{
	train_l <- c()
	train_l1 <- c()

	  # Training batch loop
	  coro::loop(for (batch in train_dl) {
	
		#batch=NULL;x=torch::torch_tensor(X[1:32,]);batch$X=x;batch$Y=x
	
		# Zero gradients 
		optimizer$zero_grad()

		# Forward pass
		output <- net(batch$X$to(device = device, non_blocking = TRUE))
		#Vx <- which(colSums(A) == 0)
		#output[ ,Vx] <- batch$X[ ,Vx]

		# Reconstruction loss
		loss <- loss.fn(output, batch$Y$to(device = device, non_blocking = TRUE))

		# Backward pass
		loss$backward()
		# Apply gradient masks to all masked layers
		for (module in net$modules) {
		 if (inherits(module, "MaskedLinear")) {
			module$apply_gradient_mask()
		 }
		}

		# Update weights
		optimizer$step()
		# Reapply weight mask to ensure weights stay zero
		torch::with_no_grad({
		  for (module in net$modules) {
		   if (inherits(module, "MaskedLinear")) {
			 module$weight$mul_(module$mask)
		   }
		  }
		})
		
		train_l <- c(train_l, as.numeric(loss))
		train_l1 <- c(train_l1, 0)
	  })

	return(list(output = output, train_l = train_l, train_l1 = train_l1))
}

batch_loop_GAE <- function(train_dl, net, optimizer, loss.fn, device, ...)
{
	train_l <- c()
	train_l1 <- c()

	  # Training batch loop
	  coro::loop(for (batch in train_dl) {
	
		#batch=NULL;x=torch::torch_tensor(X[1:32,]);batch$X=x;batch$Y=x
	
		# Zero gradients 
		optimizer$zero_grad()

	    # Forward pass
		output <- net(batch$X$to(device = device, non_blocking = TRUE))
		#Vx <- which(colSums(A) == 0)
		#output[ ,Vx] <- batch$X[ ,Vx]
		
		# Reconstruction loss
		loss <- loss.fn(output$x_hat, batch$Y$to(device = device, non_blocking = TRUE))

		# DAG constraint 
		h_A <- net$h_func()

		# L1 regularization for sparsity
		l1_reg <- torch::torch_sum(torch::torch_abs(output$W_hat))

		# Total loss
		opt <- list(l_1 = 0.01, l_2 = 0.01, l_3 = 10, h_tol = 1e-8)
		total_loss <- loss + opt$l_1 * l1_reg + opt$l_2 * h_A + (opt$l_3/2) * h_A^2

		# Backward pass
		total_loss$backward()
		
		# Update weights
		optimizer$step() 

		train_l <- c(train_l, as.numeric(loss))
		train_l1 <- c(train_l1, as.numeric(opt$l_1*l1_reg))
	  })

	return(list(output = output, train_l = train_l, train_l1 = train_l1))
}

batch_loop_NGM <- function(train_dl, net, optimizer, loss.fn, device, ...)
{
	train_l <- c()
	train_l1 <- c()
	
	  # Training batch loop through NGM
	  coro::loop(for (batch in train_dl) {

		#batch=NULL;x=torch::torch_tensor(X[1:32,]);batch$X=x;batch$Y=x

		# Zero gradients
		optimizer$zero_grad()
		
		# Forward pass
		output <- net(batch$X$to(device = device, non_blocking = TRUE))
		#Vx <- which(colSums(A) == 0)
		#output[ ,Vx] <- batch$X[ ,Vx]
		
		# Reconstruction loss
		loss <- loss.fn(output$x_hat, batch$Y$to(device = device, non_blocking = TRUE))

		# L1 constraint for DAG matching 
		l1 <- torch::torch_norm(output$Wa, p=1) # l1
		p <-  output$Wa$shape[2] # p
		lambda <- torch::torch_norm(output$Wa, p=2)/(p^2) # lambda
		l1_loss <- lambda * torch::torch_log(l1)
		#magnitude <- floor(log10(as.numeric(l1)))
		#lambda <- 10^(-magnitude)
		#l1_loss <- lambda * torch::torch_log(l1)
		
		# Total loss
		total_loss <- loss + l1_loss
		  
		# Backward pass
		total_loss$backward()
		
		# Update weights
		optimizer$step()
		
		train_l <- c(train_l, as.numeric(loss))
		train_l1 <- c(train_l1, as.numeric(l1_loss))
	  })
  
	return(list(output = output, train_l = train_l, train_l1 = train_l1))
}

extract_fitted_values <- function(object, newdata = NULL, ...) {
	# Set the model to evaluation mode
	algo <- object$param$algo
	model <- object$net # model
	vx <- colnames(object$data$X) # vx
	vy <- colnames(object$data$Y) # vy
	
	model$eval()

	# Convert data in torch_tensor
	if (is.null(newdata)) {
	 x <- torch::torch_tensor(as.matrix(object$data$X), dtype = torch::torch_float(), device = "cpu")
	} else {
	 x <- torch::torch_tensor(as.matrix(newdata[ ,vx]), dtype = torch::torch_float(), device = "cpu")
	}

	# Disable gradient calculations
	torch::with_no_grad({
	 fit <- model(x) #str(fit)
	})
	
	# Return results in R
	if (algo == "neuralgraph") {
	 y_hat <- as.matrix(fit$x_hat)
	 colnames(y_hat) <- vy
	 W_hat <- NULL #Wa?
	} else {
	 y_hat <- as.matrix(fit)
	 colnames(y_hat) <- vy
	 W_hat <- NULL #W1?
	}
	
	n <- nrow(y_hat)
	d <- ncol(y_hat)
	y_hat <- y_hat + matrix(rnorm(n * d), nrow = n, ncol = d) * 0.001
	
	return(list(y_hat = y_hat, W_hat = W_hat))
}
