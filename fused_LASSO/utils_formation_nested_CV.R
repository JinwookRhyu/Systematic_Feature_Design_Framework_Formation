# A couple of functions that are handy for testing the fused lasso and co
# (Probably this implementation is somewhat naive as I'm not an R native)
# Author and Copyright: Joachim Schaeffer, joachim.schaeffer@posteo.de
# Modified by Jinwook Rhyu, jrhyu@mit.edu

mean_sd <- function(X) {
  mean_ <- colMeans(X)
  sd <- sqrt(drop(scale(X, mean_, FALSE) ^ 2))
  return <- list(mean = mean_, sd = sd)
}

# EXPERIMENTAL FUNCTION
mean_sd_fused_lasso <- function (x, weights = rep(1, nrow(x)))
{
  weights <- weights / sum(weights)
  xm <- drop(t(weights) %*% x)
  xv <- drop(t(weights) %*% scale(x, xm, FALSE) ^ 2)
  xv[xv < 10 * .Machine$double.eps] <- 0
  list(mean = xm, sd = sqrt(xv))
}

centerXtrain <- function(X) {
  mean_ <- colMeans(X)
  X_ <- sweep(X, 2, mean_, "-")
  return <- list("X_" = X_, "mean" = mean_)
}

centerXtest <- function(X, X_train_list) {
  X_ <- sweep(X, 2, X_train_list$mean, "-")
  return <-  X_
}

standardize_y_train <- function(y) {
  N <- dim(y)[1]
  if (is.null(N)) {
    N <- size(y)[2]
  }
  mean_y <- mean(y)
  std_y <- sd(y)
  y_std_ <- (y - mean_y) / std_y
  return <-
    list(
      "y_std" = y_std_,
      "mean" = mean_y,
      "std" = std_y,
      "dim" = N
    )
}

standardize_y_test <- function(y, y_train_list) {
  mean_y_train <- y_train_list$mean
  std_y_train <- y_train_list$std
  return <- (y - mean_y_train) / std_y_train
}

rescale_y <- function(y, y_train_list) {
  mean <- y_train_list$mean
  sd_ <- y_train_list$std
  return <-  (y * sd_) + mean
}

plot_predictions_formation <-
  function(y_train_std,
           y_pred,
           y_test_std,
           y_pred_test,
           y_train_list,
           if_logtransform) {
    
    if (if_logtransform == TRUE) {
      train_df <-
      setNames(data.frame(exp(y_train_std), exp(rescale_y(
        y_pred, y_train_list
      ))),
      c("y_train", "y_train_pred"))
    } else {
      train_df <-
      setNames(data.frame(y_train_std, rescale_y(
        y_pred, y_train_list
      )),
      c("y_train", "y_train_pred"))
    }
    err_train = rmserr(train_df$y_train, train_df$y_train_pred)
    print(err_train$rmse)
    mape_train = mean(abs((train_df$y_train-train_df$y_train_pred)/train_df$y_train)) * 100
    print(mape_train)

    if (if_logtransform == TRUE) {
      test_df <-
      setNames(data.frame(exp(y_test_std), exp(rescale_y(
        y_pred_test, y_train_list
      ))), c("y_test", "y_test_pred"))
    } else {
      test_df <-
      setNames(data.frame(y_test_std, rescale_y(
        y_pred_test, y_train_list
      )), c("y_test", "y_test_pred"))
    }
    err_test = rmserr(test_df$y_test, test_df$y_test_pred)
    print(err_test$rmse)
    mape_test = mean(abs((test_df$y_test-test_df$y_test_pred)/test_df$y_test)) * 100
    print(mape_test)
  


    p <- ggplot() +
      geom_point(data = train_df,
                 aes(x = y_train, y = y_train_pred),
                 color = "#cc0000") +
      geom_point(data = test_df,
                 aes(x = y_test, y = y_test_pred),
                 col = "#00008B")
    p + labs(x = "y true", y = "y pred")
  }

plot_one_set_predictions <-
  function(y_train,
           y_pred,
           y_train_list) {
    
    train_df <-
      setNames(data.frame(y_train, rescale_y(y_pred, y_train_list)),
               c("y_train", "y_train_pred"))
    err_train = rmserr(train_df$y_train, train_df$y_train_pred)
    print(err_train$rmse)

    p <- ggplot() +
      geom_point(data = train_df,
                 aes(x = y_train, y = y_train_pred),
                 color = "#cc0000") +
      geom_abline(intercept = 0, slope = 1)
    p + labs(x = "y true", y = "y pred")
  }

predict_plot_formation <-
  function(trained_model,
           lambda_val,
           X_train_,
           X_test_,
           y_train_list,
           if_logtransform,
           model = "cvfit") {
    if (model == "cvfit") {
      y_pred <-
        predict(trained_model, X_train_, s = lambda_val)
      y_pred_test <-
        predict(trained_model, X_test_, s = lambda_val)
    }
    else {
      y_pred <-
        predict(trained_model,
                lambda = lambda_val,
                Xnew = X_train_)$fit
      y_pred_test <-
        predict(trained_model,
                lambda = lambda_val,
                Xnew = X_test_)$fit
    }
    if (if_logtransform == TRUE) {
    plot_predictions_formation(y_train_log,
                         y_pred,
                         y_test_log,
                         y_pred_test,
                         y_train_list,
                         if_logtransform)
    } else {
    plot_predictions_formation(y_train,
                         y_pred,
                         y_test,
                         y_pred_test,
                         y_train_list,
                         if_logtransform)
    }
  }

cv_fusedlasso <-
  function(X,
           y,
           foldid,
           if_logtransform,
           obj_fun,
           minlam = c(1e-6, 0),
           maxsteps = c(2000, 6000),
           lambda_seq = 0) {
    op <- options(nwarnings = 10000) # Keep all warnings!

    
    X_list <- centerXtrain(X)
    X_cen <- X_list$X_

    ind_col <- which(colStdevs(X_cen) == max(colStdevs(X_cen)), arr.ind = TRUE)
    X_cen <- X_cen / colStdevs(X_cen[,ind_col])

    y_log <- log(unname(as.matrix(y)))
    y_std_list <- standardize_y_train(y)
    y_log_std_list <- standardize_y_train(y_log) 
    y_std <- y_std_list$y_std
    y_log_std <- y_log_std_list$y_std

    x_cen <-
      c(rep(0, dim(X_cen)[2] - 2), 1, -1, rep(0, dim(X_cen)[2] - 1))
    D1_cen <- toeplitz2(x_cen, dim(X_cen)[2], dim(X_cen)[2])
    D1_cen <- getGraph(D1_cen)

    if (if_logtransform==TRUE){
      fusedlasso.fit <-
      fusedlasso(y_log_std,
               X_cen,
               graph=D1_cen,
               maxsteps = maxsteps[1],
               minlam = minlam[1],
               eps = 0)
      if (lambda_seq[1] == 0) {
        lambda_seq <- fusedlasso.fit$lambda
        minlam[2] = min(lambda_seq)
        disp("Using automatically determined lambda seqeunce!")
      }
      else{
        disp("Using custom lambda sequence!")
      }
    }
    else{
      fusedlasso.fit <-
      fusedlasso(y_std,
               X_cen,
               graph=D1_cen,
               maxsteps = maxsteps[1],
               minlam = minlam[1],
               eps = 0)
      if (lambda_seq[1] == 0) {
        lambda_seq <- fusedlasso.fit$lambda
        minlam[2] = min(lambda_seq)
        disp("Using automatically determined lambda seqeunce!")
      }
      else{
        disp("Using custom lambda sequence!")
      }
    }

    ## Evaluate each lambda on each fold:
    fold.lambda.losses <-
      tapply(seq_along(foldid), foldid, function(fold.indices) {
        id_inner = foldid[fold.indices[1]]
        X_train <- X[-fold.indices,]
        X_test <- X[fold.indices,]
        X_train_cen_list <- centerXtrain(X_train)
        X_train_cen <- X_train_cen_list$X_

        ind_col <- which(colStdevs(X_train_cen) == max(colStdevs(X_train_cen)), arr.ind = TRUE)
        
        rescale_factor = colStdevs(X_train_cen[,ind_col])
        X_train <- X_train / rescale_factor
        X_test <- X_test / rescale_factor
        
        X_train_cen_list <- centerXtrain(X_train)
        X_train_cen <- X_train_cen_list$X_
        X_test_cen <- centerXtest(X_test, X_train_cen_list)

        y_train <- unname(as.matrix(y[-fold.indices]))
        y_test <- unname(as.matrix(y[fold.indices]))
        y_train_log <- log(unname(as.matrix(y[-fold.indices])))
        y_test_log <- log(unname(as.matrix(y[fold.indices])))
        y_train_std_list <- standardize_y_train(y_train)
        y_train_log_std_list <- standardize_y_train(y_train_log)
        y_train_std <- y_train_std_list$y_std
        y_train_log_std <- y_train_log_std_list$y_std
        

        x_cen_fold <-
          c(rep(0, dim(X_train)[2] - 2), 1, -1, rep(0, dim(X_train)[2] - 1))
        D1_cen_fold <- toeplitz2(x_cen_fold, dim(X_train)[2], dim(X_train)[2])
        D1_cen_fold <- getGraph(D1_cen_fold)

        disp("step")
        fold.fusedlasso.fit <- fusedlasso(
          y = y_train_log_std,
          X = X_train_cen,
          graph=D1_cen_fold,
          maxsteps = maxsteps[2],
          minlam = minlam[2],
          eps = 0
        )
        print(min(fold.fusedlasso.fit$lambda))
        fold.fusedlasso.preds <- predict(fold.fusedlasso.fit,
                                       lambda = lambda_seq[lambda_seq > min(fold.fusedlasso.fit$lambda)], Xnew = X_test_cen)$fit
        fold.fusedlasso.betas <- coef(fold.fusedlasso.fit, lambda = lambda_seq[lambda_seq > min(fold.fusedlasso.fit$lambda)])$beta
        lambda.losses <- rep(1e6, length(lambda_seq))
        len_ind <- dim(fold.fusedlasso.preds)[2]
        
        if (if_logtransform == TRUE) {
          if (obj_fun == "MAPE"){
          lambda.losses[1:len_ind] <-
          colMeans(abs((exp(
            rescale_y(fold.fusedlasso.preds, y_train_log_std_list)
          )
          - matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE)) / (matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE)) )) * 100
          } else if (obj_fun == "RMSE"){
          lambda.losses[1:len_ind] <-
          sqrt(colMeans((exp(
            rescale_y(fold.fusedlasso.preds, y_train_log_std_list)
          )
          - matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE) )^2))
          }

          lm_df <- as.data.frame(fold.fusedlasso.betas)
          names(lm_df) <- lambda_seq[1:len_ind]
          write.csv(lm_df,
                  paste(
                    path_base,
                    paste(paste(paste(paste(paste(paste("/results_fusedlasso/log_",data_type, sep = ""),"_betamatrix_outer", sep = ""), id_outer, sep = ""), "_inner", sep=""), id_inner, sep=""), ".csv", sep = ""),
                    sep = ""
                  ))

        } else {
          if (obj_fun == "MAPE"){
          lambda.losses[1:len_ind] <-
          colMeans(abs((
            rescale_y(fold.fusedlasso.preds, y_train_std_list)
          - matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE)) / (matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE)) )) * 100
          } else if (obj_fun == "RMSE"){
          lambda.losses[1:len_ind] <-
          sqrt(colMeans((
            rescale_y(fold.fusedlasso.preds, y_train_std_list)
          - matrix(y_test, nrow=length(y_test), ncol = len_ind, byrow=FALSE))^2))
          }
          lm_df <- as.data.frame(fold.fusedlasso.betas)
          names(lm_df) <- lambda_seq[1:len_ind]
          write.csv(lm_df,
                  paste(
                    path_base,
                    paste(paste(paste(paste(paste(paste("/results_fusedlasso/",data_type, sep = ""),"_betamatrix_outer", sep = ""), id_outer, sep = ""), "_inner", sep=""), id_inner, sep=""), ".csv", sep = ""),
                    sep = ""
                  ))
        }

        return (lambda.losses)
      })
    # CV loss for each lambda:
    cv.lossmatrix <- do.call(rbind, fold.lambda.losses)
    return <-
      list(
        "fusedlasso.fit" = fusedlasso.fit,
        "lossmatrix" = cv.lossmatrix,
        "lambda_vals" = lambda_seq
      )
  }
