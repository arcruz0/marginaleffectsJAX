library(marginaleffects)
mod <- lm(mpg ~ hp, mtcars)
preds_raw <- predictions(mod)

enable_JAX_backend()
preds_jax <- predictions(mod)

expect_equal(preds_jax$estimate, preds_raw$estimate)
expect_equal(preds_jax$std.error, preds_raw$std.error)