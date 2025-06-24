#' JAX backend
#'
#' Sets up a JAX backend for `marginaleffects`, using reticulate
#' @export
#' @examples
#' library(marginaleffects)
#' library(marginaleffectsJAX)
#' initialize_JAX_backend()
#' mod <- lm(mpg ~ hp + am, data = mtcars)
#' head(predictions(mod))

initialize_JAX_backend <- function() {
  # require Python libraries
  reticulate::py_require("numpy")
  reticulate::py_require("jax")
  reticulate::py_require("functools")

  # load Python libraries to "mej_env" environment
  mej_env$np  <- reticulate::import("numpy")
  mej_env$jax <- reticulate::import("jax")
  mej_env$jnp <- reticulate::import("jax.numpy")
  mej_env$functools <- reticulate::import("functools")

  # JIT and Jacobian functions

  # predictions
  mej_env$get_predict_lm <- mej_env$jax$jit(
    function(coefs, X) {
      return(mej_env$jnp$matmul(X, coefs))
    }
  )

  mej_env$j_get_predict_lm <- mej_env$jax$jacfwd(mej_env$get_predict_lm)

  # average predictions (TODO)

  # mej_env$jit_partial <- mej_env$functools$partial(mej_env$jax$jit, static_argnums = 3L)
  #
  # mej_env$get_avg_predict_lm <- mej_env$jit_partial(
  #   function(coefs, X, by, num_groups){
  #     preds <- mej_env$jnp$matmul(X, coefs)
  #     # create groups
  #     g <- mej_env$jnp$unique(by, return_inverse = T, size = num_groups)[[2]]
  #     # get group means
  #     preds_sum <- mej_env$jax$ops$segment_sum(preds, g, num_segments = num_groups)
  #     # get group counts
  #     preds_count = mej_env$jnp$bincount(g, length = num_groups)
  #     # calculate means
  #     out = preds_sum / preds_count
  #     return(out)
  #   }
  # )
  #
  # mej_env$j_get_avg_predict_lm <- mej_env$jax$jacfwd(mej_env$get_avg_predict_lm)

  # set option and notify user
  options(marginaleffects_jacobian_function = marginaleffectsJAX:::jax_jacobian)
  insight::format_message(
    "JAX set as backend for `marginaleffects`.",
    "Run `options(marginaleffects_jacobian_function = NULL)` to disable."
  ) |>
    message()
}

mej_env <- new.env(parent = emptyenv())
