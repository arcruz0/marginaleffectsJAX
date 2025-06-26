# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5

test_predictions <- function(expr_preds, tolerance = tol){
  enable_JAX_backend()
  preds_jax <- eval(expr_preds)
  disable_JAX_backend()
  preds_no_jax <- eval(expr_preds)
  
  expect_equal(preds_jax$estimate, preds_no_jax$estimate, tolerance = tolerance)
  expect_equal(preds_jax$std.error, preds_no_jax$std.error, tolerance = tolerance)
}

library(marginaleffects)
mod <- lm(mpg ~ hp + am, mtcars)

# predictions() ----

test_predictions(
  predictions(mod)
)

# predictions(, by = F) ----

test_predictions(
  predictions(mod, by = F)
)

# predictions(, by = T) ----

test_predictions(
  predictions(mod, by = T)
)

# avg_predictions() ----

test_predictions(
  avg_predictions(mod)
)

# predictions(, by = var) ----

## dummy

test_predictions(
  predictions(mod, by = "am")
)

## character

mod_factor <- lm(bill_len ~ bill_dep + species, penguins)

test_predictions(
  predictions(mod_factor, by = "species")
)

