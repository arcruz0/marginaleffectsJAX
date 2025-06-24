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
