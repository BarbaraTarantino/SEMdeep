
# Example from getShapleyR2 with CRAN versions of packages


install.packages("shapr")
install.packages("SEMdeep")

library(SEMdeep)
library(shapr)




ig<- alsData$graph
data<- alsData$exprs
data<- transformData(data)$data

#...with train-test (0.5-0.5) samples
set.seed(123)
train<- sample(1:nrow(data), 0.5*nrow(data))
rf0<- SEMml(ig, data, train=train, algo="rf", vimp=FALSE)

res<- getShapleyR2(rf0, data[-train, ], thr=NULL, verbose=TRUE)

saveRDS(res,file = "shapr_SEMdeep_cran_test.rds")


####
#### Modified script with GitHub development version ####

remotes::install_github("NorskRegnesentral/shapr")
#remotes::install_github("martinju/SEMdeep",ref = "fix_new_shapr_PR")
devtools::load_all()



ig<- alsData$graph
data<- alsData$exprs
data<- transformData(data)$data

#...with train-test (0.5-0.5) samples
set.seed(123)
train<- sample(1:nrow(data), 0.5*nrow(data))
rf0<- SEMml(ig, data, train=train, algo="rf", vimp=FALSE)

res<- getShapleyR2(rf0, data[-train, ], thr=NULL, verbose=TRUE)

saveRDS(res,file = "shapr_SEMdeep_GH_test.rds")






#### Checking differences ##########


res_cran <- readRDS(file = "shapr_SEMdeep_cran_test.rds")
res_gh <- readRDS(file = "shapr_SEMdeep_GH_test.rds")

comp <- cbind(res_cran$est$sign_r2,res_gh$est$sign_r2)
comp <- cbind(comp,comp[,1]-comp[,2])

head(comp)
# Just minor differences
#[,1]        [,2]         [,3]
#[1,]  0.29525926  0.30137196 -0.006112700
#[2,]  0.04201096  0.05096058 -0.008949624
#[3,]  0.22190555  0.22485745 -0.002951901
#[4,]  0.11511283  0.10610216  0.009010672
#[5,]  0.00000000  0.00000000  0.000000000
#[6,] -0.28799528 -0.24696402 -0.041031264

mean(abs(comp[,3]))
#[1] 0.01341321




