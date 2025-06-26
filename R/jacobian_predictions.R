jacobian_predictions <- function(coefs, X){
  mej_env$j_get_predict_lm(mej_env$np$array(coefs), X) |>
    mej_env$np$array()
}

jacobian_predictions_byT <- function(coefs, X){
  mej_env$j_get_predict_byT_lm(mej_env$np$array(coefs), X) |>
    mej_env$np$array() |> 
    as.matrix() |> 
    t()
}

jacobian_predictions_by_var <- function(coefs, X, by_index, num_groups){
  mej_env$j_get_predict_by_var_lm(
    mej_env$np$array(coefs), X, mej_env$np$array(by_index), num_groups
  ) |>
    mej_env$np$array()
}
