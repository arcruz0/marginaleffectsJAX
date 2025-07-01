jax_jacobian <- function(coefs, newdata, model, hypothesis, calling_function, ...) {
  # testing only:
  # list_args <<- c(as.list(environment()), list(...))
  # message("Wrote `list_args` to Global environment")
  
  # https://stackoverflow.com/a/17244041
  list_args <- c(as.list(environment()), list(...))
  
  verbose <- getOption("marginaleffectsJAX_verbose", default = FALSE)
  
  if (isTRUE(verbose)) message("--Calling JAX backend")
  
  if (!identical(calling_function, "predictions")) {
    msg <- "`marginaleffectsJAX` only supports predictions(). Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }

  if (!is.null(hypothesis) && !identical(calling_function, "hypotheses")) {
    msg <- "`marginaleffectsJAX` does not support the `hypothesis` argument. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }

  X <- attr(newdata, "marginaleffects_model_matrix")
  if (!isTRUE(checkmate::check_matrix(X, null.ok = FALSE))) {
    msg <- "`marginaleffectsJAX` cannot access the model matrix. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }
  # supported models
  if (!identical(class(model)[1], "lm")) { # first element of class vector because `glm` inherits from `lm`
    msg <- "`marginaleffectsJAX` only supports models of class `lm`. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }
  
  # supported predict arguments
  if (isFALSE(is.null(list_args$by) | is.logical(list_args$by) |
        is.character(list_args$by) | is.data.frame(list_args$by))){
    msg <- "`marginaleffectsJAX` only supports NULL, a character, T/F, or a data.frame in the `by =` argument of `predictions()`. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }
  
  # executing JAX
  if (isTRUE(verbose)) message("--Executing JAX function")
  
  J <- NULL
  
  # unit-level predictions
  
  if (isTRUE(is.null(list_args$by) | isFALSE(list_args$by))){
    J <- jacobian_predictions(coefs, X)
  } 
  
  # average predictions
  
  if (isTRUE(list_args$by)) {
    J <- jacobian_predictions_byT(coefs, X)
  }
  
  # marginal predictions
  
  if (is.null(J)){
    if (is.data.frame(list_args$by)){
      # if `by` is a data.frame, merge into `newdata`
      newdata <- merge(newdata, list_args$by, sort = F, all.x = T)
      by_index <- newdata[["by"]] |> as.factor()
    } else if (is.character(list_args$by)){
      by_index <- newdata[[list_args$by]] |> as.factor()
    }

    num_groups <- nlevels(by_index)
    by_index <- by_index |> as.integer()
    num_groups <- by_index |> unique() |> length()
    J <- jacobian_predictions_by_var(coefs, X, by_index, num_groups)
  }
  
  if (isTRUE(verbose)){
    if (!is.null(J)){
      message("--Succesfully executed JAX function")
    } else {
      message("--JAX function erred. Reverting to standard computation.")
    }
  }
  
  return(J)
}
