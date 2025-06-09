# Script for Fused Lasso approach for feature design framework 
# Originally written by Joachim Schaeffer joachim.schaeffer@posteo.de (https://doi.org/10.1016/j.compchemeng.2023.108471)
# Modified by Jinwook Rhyu jrhyu@mit.edu for cross-validation

## CLEAN UP
rm(list = ls())
tryCatch(
  p_unload(all),
  error = function(e) {
    print("Skip unloading addons, probably no addons!")
  }
)
tryCatch(  
  dev.off(),
  error = function(e) {
    print("Skip clearing plots, probably no plots!")
  } 
)
cat("\014")

## Model Settings
if_logtransform_list <- list(TRUE, FALSE)
data_type_list <- list("B_Q_V", "C_Q_V", "B_V_t", "C_V_t", "A_Q_V", "A_t_V")
obj_fun <- "RMSE"

## LOADING
pacman::p_load(pacman,
               MASS,
               bayesreg,
               glmnet,
               rio, 
               ggplot2,
               pracma,
               genlasso,
               Matrix,
               resample,
               tibble)

# Please change this to the user's repository
path_base <- getwd() # Current directory
source(paste(path_base, "/utils_formation_nested_CV.R", sep = ""))
if (!dir.exists(paste(path_base, "/regression_in_R", sep = ""))) {
  dir.create(paste(path_base, "/regression_in_R", sep = ""))
}

for (id_log in 1:2){
  if_logtransform <- if_logtransform_list[id_log]
  for (id_data in 1:6){
    data_type <- data_type_list[id_data]
    set.seed(42)

    ## Load Data
    path <- paste(path_base, paste(paste("/data_formation/",data_type, sep = ""),"_all.xlsx", sep = ""), sep = "")
    formation_data = import(path)
    numcol = dim(formation_data)[2]
    

    for (id_outer in 0:4){
      ## Construct Data Matrices
      test_id_outer <- formation_data[, numcol] == id_outer
      train_id_outer <- formation_data[, numcol] != id_outer

      nfolds <- 5
      foldid <- match(formation_data[train_id_outer,(numcol-1)], sort(unique(formation_data[train_id_outer,(numcol-1)]))) %% nfolds

      X <- unname(as.matrix(rev(formation_data[train_id_outer, 2:(numcol-3)])))
      y <- formation_data[train_id_outer, (numcol-2)]
      protocols <- formation_data[train_id_outer, (numcol-1)]

      cv_list = cv_fusedlasso(X, y, foldid, if_logtransform, obj_fun)
        
        
      if (if_logtransform == TRUE) {
        lm_df <- as.data.frame(cv_list$lossmatrix)
        names(lm_df) <- cv_list$lambda_vals
        write.csv(lm_df,
                paste(
                  path_base,
                  paste(paste(paste(paste("/regression_in_R/log_",data_type, sep = ""),"_cv_lossmatrix_outer", sep = ""), id_outer, sep = ""), ".csv", sep=""),
                  sep = ""
                ))
        } else {
        lm_df <- as.data.frame(cv_list$lossmatrix)
        names(lm_df) <- cv_list$lambda_vals
        write.csv(lm_df,
                paste(
                  path_base,
                  paste(paste(paste(paste("/regression_in_R/",data_type, sep = ""),"_cv_lossmatrix_outer", sep = ""), id_outer, sep = ""), ".csv", sep=""),
                  sep = ""
                ))
    }

    }

  }
}
