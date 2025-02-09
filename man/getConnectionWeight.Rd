% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SEMdnn.R
\name{getConnectionWeight}
\alias{getConnectionWeight}
\title{Connection Weight method for neural network variable importance}
\usage{
getConnectionWeight(object, thr = NULL, verbose = FALSE, ...)
}
\arguments{
\item{object}{A neural network object from \code{SEMdnn()} function.}

\item{thr}{A numeric value [0-1] indicating the threshold to apply to the
Olden's connection weights to color the graph. If thr = NULL (default), the
threshold is set to thr = 0.5*max(abs(connection weights)).}

\item{verbose}{A logical value. If FALSE (default), the processed graph 
will not be plotted to screen.}

\item{...}{Currently ignored.}
}
\value{
A list of three object: (i) est: a data.frame including the connections together
with their connection weights(W), (ii) gest: if the outcome vector is given, a data.frame
of connection weights for outcome lavels, and (iii) dag: DAG with colored edges/nodes. If
abs(W) > thr and W < 0, the edge W > 0, the edge is activated and it is highlighted
in red. If the outcome vector is given, nodes with absolute connection weights summed
over the outcome levels, i.e. sum(abs(W[outcome levels])) > thr, will be highlighted
in pink.
}
\description{
The function computes the matrix multiplications of hidden
weight matrices (Wx,...,Wy), i.e., the product of the raw input-hidden and
hidden-output connection weights between each input and output neuron and
sums the products across all hidden neurons, as proposed by Olden (2004).
}
\details{
In a neural network, the connections between inputs and outputs are 
represented by the connection weights between the neurons. 
The importance values assigned to each input variable using the Olden method are
in units that are based directly on the summed product of the connection weights.
The amount and direction of the link weights largely determine the proportional
contributions of the input variables to the neural network's prediction output.
Input variables with larger connection weights indicate higher intensities
of signal transfer and are therefore more important in the prediction process.
Positive connection weights represent excitatory effects on neurons (raising the
intensity of the incoming signal) and increase the value of the predicted response, 
while negative connection weights represent inhibitory effects on neurons 
(reducing the intensity of the incoming signal). The weights that change sign
(e.g., positive to negative) between the input-hidden to hidden-output layers
would have a cancelling effect, and vice versa weights with the same sign would
have a synergistic effect.
Note that in order to map the connection weights to the DAG edges, the element-wise
product, W*A is performed between the Olden's weights entered in a matrix, W(pxp)
and the binary (1,0) adjacency matrix, A(pxp) of the input DAG.
}
\examples{

\donttest{
if (torch::torch_is_installed()){

# load ALS data
ig<- alsData$graph
data<- alsData$exprs
data<- transformData(data)$data

#ncores<- parallel::detectCores(logical = FALSE)
dnn0<- SEMdnn(ig, data, outcome = NULL, thr = NULL,
			#hidden = 5*K, link = "selu", bias = TRUE,
			hidden = c(10,10,10), link = "selu", bias = TRUE,
			validation = 0,  epochs = 32, ncores = 2)

cw05<- getConnectionWeight(dnn0, thr = 0.5, verbose = TRUE)
table(E(cw05$dag)$color)
}
}

}
\references{
Olden, Julian & Jackson, Donald. (2002). Illuminating the "black box": A 
randomization approach for understanding variable contributions in artificial
neural networks. Ecological Modelling. 154. 135-150. 10.1016/S0304-3800(02)00064-9. 

Olden, Julian. (2004). An accurate comparison of methods for quantifying 
variable importance in artificial neural networks using simulated data. 
Ecological Modelling. 178. 10.1016/S0304-3800(04)00156-5.
}
\author{
Mario Grassi \email{mario.grassi@unipv.it}
}
