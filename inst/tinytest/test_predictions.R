library(marginaleffects)

# Define tests

# Set tolerance - autodiff may be more precise than numerical differentiation
tol <- 1e-6

# Test predictions() calls
test_predictions <- function(expr_preds, tolerance = tol){
  enable_JAX_backend(verbose = T)
  expect_message(eval(expr_preds), "Succesfully executed JAX function")
  preds_jax <- eval(expr_preds)
  disable_JAX_backend()
  preds_no_jax <- eval(expr_preds)
  
  expect_equal(preds_jax$estimate, preds_no_jax$estimate, tolerance = tolerance)
  expect_equal(preds_jax$std.error, preds_no_jax$std.error, tolerance = tolerance)
}

# Test plot_predictions() calls
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

# Scramble dataset to make sure order doesn't matter
set.seed(1)
penguins2 <- penguins[sample(1:nrow(penguins), nrow(penguins)),]

# Create dummy/integer variable
penguins2$dummy_female <- ifelse(penguins2$sex == "female", 1L, 0L)

# Estimate baseline model for tests
mod <- lm(bill_len ~ bill_dep + dummy_female, penguins2)



# predictions() ----------------------------------------------------------------

test_predictions(
  predictions(mod)
)

test_predictions(
  predictions(mod, by = F)
)



# predictions(, by = T) --------------------------------------------------------

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
  predictions(mod, by = "dummy_female")
)

test_predictions(
  avg_predictions(mod, by = "dummy_female")
)

test_plot_predictions(
  plot_predictions(mod, by = "dummy_female")
)

## character

mod_chr <- lm(bill_len ~ bill_dep + species_chr,
               penguins2 |> transform(species_chr = as.character(species)))

test_predictions(
  predictions(mod_chr, by = "species_chr")
)

test_plot_predictions(
  plot_predictions(mod_chr, by = "species_chr")
)

## factor: pre-existing factor variable

mod_factor_pre <- lm(bill_len ~ bill_dep + species, penguins2)

test_predictions(
  predictions(mod_factor_pre, by = "species")
)

test_plot_predictions(
  plot_predictions(mod_factor_pre, by = "species")
)

## factor: character variable passed as factor()

mod_factor_from_chr <- lm(
  bill_len ~ bill_dep + factor(species_chr), 
  penguins2 |> transform(species_chr = as.character(species))
)

test_predictions(
  predictions(mod_factor_from_chr, by = "species_chr")
)

test_plot_predictions(
  plot_predictions(mod_factor_from_chr, by = "species_chr")
)

## factor: integer variable passed as factor()
mod_factor_from_int <-  lm(bill_len ~ bill_dep + factor(year), penguins2)

test_predictions(
  predictions(mod_factor_from_int, by = "year")
)

test_plot_predictions(
  plot_predictions(mod_factor_from_int, by = "year")
)

## factor: numeric variable passed as factor()
mod_factor_from_num <- lm(
  bill_len ~ bill_dep + factor(year_dbl), 
  penguins2 |> transform(year_dbl = as.numeric(year))
)

test_predictions(
  predictions(mod_factor_from_num, by = "year_dbl")
)

test_plot_predictions(
  plot_predictions(mod_factor_from_num, by = "year_dbl")
)

## factor: variable passed as as.factor()
mod_factor_from_chr_as.factor <- lm(
  bill_len ~ bill_dep + as.factor(species_chr), 
  penguins2 |> transform(species_chr = as.character(species))
)

test_predictions(
  predictions(mod_factor_from_chr_as.factor, by = "species_chr")
)

test_plot_predictions(
  plot_predictions(mod_factor_from_chr_as.factor, by = "species_chr")
)



# predictions(, newdata = datagrid()) ------------------------------------------

mod_plus <- lm(bill_len ~ bill_dep + flipper_len * sex, penguins2)

## setting values

test_predictions(
  predictions(mod_plus,
              datagrid(flipper_len = 200:201, sex = c("male", "female")))
)

## setting functions

test_predictions(
  predictions(mod_plus,
              datagrid(flipper_len = mean, sex = unique))
)

## setting values + functions

test_predictions(
  predictions(mod_plus,
              datagrid(flipper_len = 200:201, sex = unique))
)



# predictions(, newdata = datagrid(, grid_type = "counterfactual")) ------------

test_predictions(
  predictions(mod_plus,
              datagrid(flipper_len = 200:201, sex = unique, 
                       grid_type = "counterfactual"))
)

test_predictions(
  predictions(mod_plus,
              variables = list(flipper_len = 200:201, sex = unique))
)

test_predictions(
  avg_predictions(
    mod_plus,
    variables = list(flipper_len = 200:201, sex = unique),
    by = "sex"
  )
)



# predictions(, newdata = "mean") ----------------------------------------------

test_predictions(
  predictions(mod_plus, newdata = "mean")
)



# predictions(, newdata = "balanced") ----------------------------------------------

test_predictions(
  predictions(mod_plus, newdata = "balanced")
)

test_predictions(
  avg_predictions(mod_plus, newdata = "balanced", by = "sex")
)



# plot_predictions(, condition = var) ------------------------------------------

test_plot_predictions(
  plot_predictions(mod_plus, condition = "bill_dep")
)

test_plot_predictions(
  plot_predictions(mod_plus, condition = c("bill_dep", "flipper_len"))
)

test_plot_predictions(
  plot_predictions(mod_plus, condition = c("bill_dep", "flipper_len", "sex"))
)
