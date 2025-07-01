# occasionally autodiff is more precise than numerical differentiation.
# must allow some tolerance
tol <- 1e-5

test_plot_predictions <- function(expr_plot_preds, tolerance = tol){
  enable_JAX_backend(verbose = T)
  expect_message(eval(expr_plot_preds), "Succesfully executed JAX function")
  plot_preds_jax <- eval(expr_plot_preds)
  disable_JAX_backend()
  plot_preds_no_jax <- eval(expr_plot_preds)
  
  expect_equal(ggplot2::ggplot_build(plot_preds_jax)$data, 
               ggplot2::ggplot_build(plot_preds_no_jax)$data, 
               tolerance = tolerance)
}

library(marginaleffects)

# Scramble datasets to make sure order doesn't make a difference
mtcars2 <- mtcars[sample(1:nrow(mtcars), nrow(mtcars)),]
penguins2 <- penguins[sample(1:nrow(penguins), nrow(penguins)),]

mod <- lm(mpg ~ hp + am, mtcars2)

# plot_predictions(, by = var) ----

## dummy

test_plot_predictions(
  plot_predictions(mod, by = "am")
)

## character

mod_chr <- lm(bill_len ~ bill_dep + species_chr, 
              penguins2 |> transform(species_chr = as.character(species)))

test_plot_predictions(
  plot_predictions(mod_chr, by = "species_chr")
)

## factor: pre-existing factor variable

mod_factor_pre <- lm(bill_len ~ bill_dep + species, penguins2)

test_plot_predictions(
  plot_predictions(mod_factor_pre, by = "species")
)

## factor: character variable passed as factor()

mod_factor_from_chr <- lm(
  bill_len ~ bill_dep + factor(species_chr), 
  penguins2 |> transform(species_chr = as.character(species))
)

test_plot_predictions(
  plot_predictions(mod_factor_from_chr, by = "species_chr")
)

# factor: integer variable passed as factor()
mod_factor_from_int <-  lm(bill_len ~ bill_dep + factor(year), penguins2)

test_plot_predictions(
  plot_predictions(mod_factor_from_int, by = "year")
)

# factor: numeric variable passed as factor()
mod_factor_from_num <- lm(mpg ~ hp + factor(am), mtcars2)

test_plot_predictions(
  plot_predictions(mod_factor_from_num, by = "am")
)

# factor: variable passed as as.factor()
mod_factor_from_num_as.factor <- lm(mpg ~ hp + as.factor(am), mtcars2)

test_plot_predictions(
  plot_predictions(mod_factor_from_num_as.factor, by = "am")
)
