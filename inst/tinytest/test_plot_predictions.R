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

# Scramble dataset to make sure order doesn't make a difference
set.seed(1)
penguins2 <- penguins[sample(1:nrow(penguins), nrow(penguins)),]
penguins2$dummy_female <- ifelse(penguins2$sex == "female", 1L, 0L)

mod <- lm(bill_len ~ bill_dep + dummy_female, penguins2)

# plot_predictions(, by = var) ----

## dummy

test_plot_predictions(
  plot_predictions(mod, by = "dummy_female")
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
mod_factor_from_num <- lm(bill_len ~ bill_dep + factor(bill_dep), penguins2)

test_plot_predictions(
  plot_predictions(mod_factor_from_num, by = "bill_dep")
)

# factor: variable passed as as.factor()
mod_factor_from_num_as.factor <- lm(bill_len ~ bill_dep + as.factor(bill_dep), penguins2)

test_plot_predictions(
  plot_predictions(mod_factor_from_num_as.factor, by = "bill_dep")
)
