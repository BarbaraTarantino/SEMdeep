% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SEMhelp.R
\name{mapGraph}
\alias{mapGraph}
\title{Map additional variables (nodes) to a graph object}
\usage{
mapGraph(graph, type = "outcome", LV = NULL, f = NULL, ...)
}
\arguments{
\item{graph}{An igraph object.}

\item{type}{A character value specifying the type of mapping. Five 
types can be specified. If \code{type = "source"} is specified, an 
additional source node (or more) is added to the graph. If 
\code{type = "group"}, an additional group source node is added. If
\code{type = "outcome"} (default), a prediction sink node is mapped
to the graph. If \code{type = "LV"}, a LV source node is included (where
the number of LV depends on the LV argument). If \code{type = "clusterLV"},
a series of clusters for the data are computed and a different LV source
node is added separately for each cluster.}

\item{LV}{The number of LV source nodes to add to the graph. This argument 
needs to be specified when \code{type = "LV"}. When \code{type = "clusterLV"}
the LV number is defined internally equal to the number of clusters.
(default = NULL).}

\item{f}{A formula object (default = NULL). A new graph object is created
according to the specified formula object.}

\item{...}{Currently ignored.}
}
\value{
mapGraph returns invisibly the graphical object with the
mapped node variables.
}
\description{
The function insert additional nodes to a graph object.
Among the node types, additional source or sink nodes can be added. 
Regarding the former, source nodes can represent: (i) data variables; 
(ii) a group variable; (iii) Latent Variables (LV). For the latter, an 
outcome variable, representing the prediction of interest, can be added. 
Moreover, \code{mapGraph()} can also create a new graph object starting
from a compact symbolic formula.
}
\examples{

# Load Amyotrophic Lateral Sclerosis (ALS)
ig<- alsData$graph; gplot(ig)

# ... map source nodes to ALS graph 
ig1 <- mapGraph(ig, type = "source"); gplot(ig1, l="dot")

# ... map group source node to ALS graph 
ig2 <- mapGraph(ig, type = "group"); gplot(ig2, l="fdp")

# ... map outcome sink to ALS graph 
ig3 <- mapGraph(ig, type = "outcome"); gplot(ig3, l="dot")

# ... map LV source nodes to ALS graph 
ig4 <- mapGraph(ig, type = "LV", LV = 3); gplot(ig4, l="fdp")

# ... map LV source nodes to the clusters of ALS graph 
ig5 <- mapGraph(ig, type = "clusterLV"); gplot(ig5, l="dot")

# ... create a new graph with the formula variables
formula <- as.formula("z4747 ~ z1432 + z5603 + z5630")
ig6 <- mapGraph(f=formula); gplot(ig6)
}
\author{
Mario Grassi \email{mario.grassi@unipv.it}
}
