# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5

test_predictions <- function(expr_preds, tolerance = tol){
  enable_JAX_backend(verbose = T)
  expect_message(eval(expr_preds), "Succesfully executed JAX function")
  preds_jax <- eval(expr_preds)
  disable_JAX_backend()
  preds_no_jax <- eval(expr_preds)
  
  expect_equal(preds_jax$estimate, preds_no_jax$estimate, tolerance = tolerance)
  expect_equal(preds_jax$std.error, preds_no_jax$std.error, tolerance = tolerance)
}

library(marginaleffects)

# Scramble datasets to make sure order doesn't make a difference
mtcars2 <- mtcars[sample(1:nrow(mtcars), nrow(mtcars)),]
penguins2 <- penguins[sample(1:nrow(penguins), nrow(penguins)),]

mod <- lm(mpg ~ hp + am, mtcars2)

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

mod_chr <- lm(bill_len ~ bill_dep + species_chr, 
               penguins2 |> transform(species_chr = as.character(species)))

test_predictions(
  predictions(mod_chr, by = "species_chr")
)

## factor: pre-existing factor variable

mod_factor_pre <- lm(bill_len ~ bill_dep + species, penguins2)

test_predictions(
  predictions(mod_factor_pre, by = "species")
)

## factor: character variable passed as factor()

mod_factor_from_chr <- lm(
  bill_len ~ bill_dep + factor(species_chr), 
  penguins2 |> transform(species_chr = as.character(species))
)

test_predictions(
  predictions(mod_factor_from_chr, by = "species_chr")
)

# factor: integer variable passed as factor()
mod_factor_from_int <-  lm(bill_len ~ bill_dep + factor(year), penguins2)

test_predictions(
  predictions(mod_factor_from_int, by = "year")
)

# factor: numeric variable passed as factor()
mod_factor_from_num <- lm(mpg ~ hp + factor(am), mtcars2)

test_predictions(
  predictions(mod_factor_from_num, by = "am")
)

# factor: variable passed as as.factor()
mod_factor_from_num_as.factor <- lm(mpg ~ hp + as.factor(am), mtcars2)

test_predictions(
  predictions(mod_factor_from_num_as.factor, by = "am")
)
