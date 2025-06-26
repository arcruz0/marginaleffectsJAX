# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5


library(marginaleffects)
mod <- lm(mpg ~ hp + am, mtcars)

# predictions() ----

enable_JAX_backend()
preds_jax <- predictions(mod)

disable_JAX_backend()
preds_raw <- predictions(mod)

expect_equal(preds_jax$estimate, preds_raw$estimate, tolerance = tol)
expect_equal(preds_jax$std.error, preds_raw$std.error, tolerance = tol)

# predictions(, by = T) ----

enable_JAX_backend()
preds_byT_jax <- predictions(mod, by = T)

disable_JAX_backend()
preds_byT_raw <- predictions(mod, by = T)

expect_equal(preds_byT_jax$estimate, preds_byT_raw$estimate, tolerance = tol)
expect_equal(preds_byT_jax$std.error, preds_byT_raw$std.error, tolerance = tol)

# avg_predictions() ----

enable_JAX_backend()
avg_preds_jax <- avg_predictions(mod)

disable_JAX_backend()
avg_preds_raw <- avg_predictions(mod)

expect_equal(avg_preds_jax$estimate, avg_preds_raw$estimate, tolerance = tol)
expect_equal(avg_preds_jax$std.error, avg_preds_raw$std.error, tolerance = tol)

# predictions(, by = var) ----

## dummy

enable_JAX_backend()
preds_by_var_dummy_jax <- predictions(mod, by = "am")

disable_JAX_backend()
preds_by_var_dummy_raw <- predictions(mod, by = "am")

expect_equal(preds_by_var_dummy_jax$estimate, 
             preds_by_var_dummy_raw$estimate, tolerance = tol)
expect_equal(preds_by_var_dummy_jax$std.error, 
             preds_by_var_dummy_raw$std.error, tolerance = tol)

## character

mod_factor <- lm(bill_len ~ bill_dep + species, penguins)

enable_JAX_backend()
preds_by_var_factor_jax <- predictions(mod_factor, by = "species")

disable_JAX_backend()
preds_by_var_factor_raw <- predictions(mod_factor, by = "species")

expect_equal(preds_by_var_factor_jax$estimate, 
             preds_by_var_factor_raw$estimate, tolerance = tol)
expect_equal(preds_by_var_factor_jax$std.error, 
             preds_by_var_factor_raw$std.error, tolerance = tol)

