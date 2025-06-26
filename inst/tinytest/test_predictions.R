# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5


library(marginaleffects)
mod <- lm(mpg ~ hp, mtcars)

# predictions() ----

preds_raw <- predictions(mod)

enable_JAX_backend()
preds_jax <- predictions(mod)

expect_equal(preds_jax$estimate, preds_raw$estimate, tolerance = tol)
expect_equal(preds_jax$std.error, preds_raw$std.error, tolerance = tol)

# predictions(, by = T) ----

preds_byT_jax <- predictions(mod, by = T)

disable_JAX_backend()
preds_byT_raw <- predictions(mod, by = T)

expect_equal(preds_byT_jax$std.error, preds_byT_raw$std.error)
expect_equal(preds_byT_jax$std.error, preds_byT_raw$std.error, tolerance = tol)

# avg_predictions() ----

avg_preds_raw <- avg_predictions(mod)

enable_JAX_backend()
avg_preds_jax <- avg_predictions(mod)

expect_equal(avg_preds_jax$estimate, avg_preds_raw$estimate, tolerance = tol)
expect_equal(avg_preds_jax$std.error, avg_preds_raw$std.error, tolerance = tol)
