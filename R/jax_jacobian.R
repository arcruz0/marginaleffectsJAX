#' Compute Jacobian using JAX
#'
#' @export
#' @examples
#' library(marginaleffects)
#' library(marginaleffectsjax)
#' mod <- lm(mpg ~ hp + am, data = mtcars)
#' predictions(mod) |> head()

jax_jacobian <- function(coefs, newdata, model, hypothesis, calling_function, ...) {
  if (!identical(calling_function, "predictions")) {
    msg <- "`marginaleffectsjax` only supports predictions(). Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }

  if (!is.null(hypothesis) && !identical(calling_function, "hypotheses")) {
    msg <- "`marginaleffectsjax` does not support the `hypothesis` argument. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }

  X <- attr(newdata, "marginaleffects_model_matrix")
  if (!isTRUE(checkmate::check_matrix(X, null.ok = FALSE))) {
    msg <- "`marginaleffectsjax` cannot access the model matrix. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }
  # supported models
  if (!identical(class(model)[1], "lm")) { # first element of class vector because `glm` inherits from `lm`
    msg <- "`marginaleffectsjax` only supports models of class `lm`. Reverting to standard computation."
    warning(msg, call. = FALSE)
    return(NULL)
  }

  J <- mej_env$j_get_predict_lm(mej_env$np$array(coefs), X) |>
    mej_env$np$array()
  return(J)
}
