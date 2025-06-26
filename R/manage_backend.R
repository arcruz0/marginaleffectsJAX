#' JAX backend
#'
#' @description
#' Manages the JAX backend for `marginaleffects`.
#' 
#' * `enable_JAX_backend()`: enables JAX backend.
#' * `disable_JAX_backend()`: disables JAX backend, optionally unloading all internal Python libraries/functions.
#' * `status_JAX_backend()`: checks the status of the JAX backend.
#' 
#' 
#' @examples
#' library(marginaleffects)
#' library(marginaleffectsJAX)
#' enable_JAX_backend()
#' mod <- lm(mpg ~ hp + am, data = mtcars)
#' head(predictions(mod)) # using JAX
#' disable_JAX_backend()
#' head(predictions(mod)) # not using JAX
#' 
#' @param verbose Whether to display informative messages any time the JAX backend is used by a `marginaleffects` function (defaults to `FALSE`).
#' 
#' @export
enable_JAX_backend <- function(verbose = FALSE) {
  # check if the environment already exists and has "jax" object
  if (!exists("jax", envir = mej_env, inherits = FALSE)){
    # require Python libraries
    reticulate::py_require("numpy")
    reticulate::py_require("jax")
    # reticulate::py_require("functools")
    
    # load Python libraries to "mej_env" environment
    mej_env$np  <- reticulate::import("numpy")
    mej_env$jax <- reticulate::import("jax")
    mej_env$jnp <- reticulate::import("jax.numpy")
    mej_env$functools <- reticulate::import("functools")
    
    # utils
    mej_env$check_jax <- function(){
      one <- mej_env$jnp$abs(-1) |> mej_env$np$array() |> as.integer()
      return(isTRUE(all.equal(one, 1L)))
    }
    
    # JIT and Jacobian functions - TODO: call from other script(s)
    
    # predictions
    mej_env$get_predict_lm <- mej_env$jax$jit(
      function(coefs, X) {
        return(mej_env$jnp$matmul(X, coefs))
      }
    )
    
    mej_env$j_get_predict_lm <- mej_env$jax$jacfwd(mej_env$get_predict_lm)
    
    # predictions(, by = T)
    mej_env$get_predict_byT_lm <- mej_env$jax$jit(
      function(coefs, X) {
        return(mej_env$jnp$mean(mej_env$jnp$matmul(X, coefs)))
      }
    )
    
    mej_env$j_get_predict_byT_lm <- mej_env$jax$jacrev(mej_env$get_predict_byT_lm)
    
    # average predictions
    mej_env$jit_partial <- mej_env$functools$partial(mej_env$jax$jit, static_argnums = 3L)

    mej_env$get_predict_by_var_lm <- mej_env$jit_partial(
      function(coefs, X, by_index, num_groups){
        preds <- mej_env$jnp$matmul(X, coefs)
        # create groups
        g <- mej_env$jnp$unique(by_index, return_inverse = T, size = num_groups)[[2]]
        # get group means
        preds_sum <- mej_env$jax$ops$segment_sum(preds, g, num_segments = num_groups)
        # get group counts
        preds_count = mej_env$jnp$bincount(g, length = num_groups)
        # calculate means
        out = preds_sum / preds_count
        return(out)
      }
    )

    mej_env$j_get_predict_by_var_lm <- mej_env$jax$jacfwd(mej_env$get_predict_by_var_lm)
  }

  # set option and notify user
  options(marginaleffects_jacobian_function = marginaleffectsJAX:::jax_jacobian)
  options(marginaleffectsJAX_verbose = verbose)
  message("JAX is now a backend for `marginaleffects`. Run `disable_JAX_backend()` to disable.")
}

#' @rdname enable_JAX_backend
#' @param hard_unload Whether to unload all Python libraries/functions (defaults to `FALSE`). Note that even after running `disable_JAX_backend(hard_unload = TRUE)`, the `reticulate` Python bindings will remain---see <https://github.com/rstudio/reticulate/issues/580#issuecomment-521364482>.
#' @export
#' 
disable_JAX_backend <- function(hard_unload = FALSE){
  if (isTRUE(hard_unload)){
    rm(list = ls(envir = mej_env), envir = mej_env)
  }
  
  # set option and notify user
  options(marginaleffects_jacobian_function = \(...) NULL)
  message("JAX is no longer a backend for `marginaleffects`. Run `enable_JAX_backend()` to reactivate.")
}

#' @rdname enable_JAX_backend
#' @export
#' 
status_JAX_backend <- function(){
  lgl_jax_status <- FALSE
  
  current_option <- getOption("marginaleffects_jacobian_function", 
                              default = \(...) NULL)
  
  if (!isTRUE(all.equal(current_option, \(...) NULL, check.environment = F))){
    lgl_jax_status <- mej_env$check_jax()
  }
  
  if (isTRUE(lgl_jax_status)){
    message("JAX is currently a backend for `marginaleffects`. Run `disable_JAX_backend()` to disable.")
    invisible(TRUE)
  } else {
    message("JAX is not currently a backend for `marginaleffects`. Run `enable_JAX_backend()` to enable.")
    invisible(FALSE)
  }
}

mej_env <- new.env(parent = emptyenv())
