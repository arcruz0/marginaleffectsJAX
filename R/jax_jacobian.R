jax_jacobian <- function(coefs, newdata, model, hypothesis, calling_function, ...) {
  # testing only:
  # list_args <<- c(as.list(environment()), list(...))
  # message("Wrote `list_args` to Global environment")
  # return(NULL)
  
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
  if (!isTRUE(checkmate::check_logical(list_args$by, null.ok = FALSE))){
    msg <- "`marginaleffectsJAX` only supports logical values in the `by =` argument of `predictions()`. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }
  
  # executing JAX
  if (isTRUE(verbose)) message("--Executing JAX function")
  J <- NULL
  
  if (isTRUE(is.null(list_args$by))){
    J <- jacobian_predictions(coefs, X)
  } else if (isTRUE(list_args$by)) {
    J <- jacobian_predictions_byT(coefs, X)
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
