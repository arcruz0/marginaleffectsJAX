# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5

library(marginaleffects)

test_plot_predictions <- function(expr_plot_preds, tolerance = tol){
  enable_JAX_backend()
  plot_preds_jax <- eval(expr_plot_preds)
  disable_JAX_backend()
  plot_preds_no_jax <- eval(expr_plot_preds)
  
  expect_equal(ggplot2::ggplot_build(plot_preds_jax)$data, 
               ggplot2::ggplot_build(plot_preds_no_jax)$data, 
               tolerance = tolerance)
}

# plot_predictions(, by = var) ----

## dummy
mod <- lm(mpg ~ hp + am, mtcars)

test_plot_predictions(
  plot_predictions(mod, by = "am")
)

## character
mod_char <- lm(bill_len ~ bill_dep + species, penguins)

test_plot_predictions(
  plot_predictions(mod_char, by = "species")
)
