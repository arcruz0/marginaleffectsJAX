# alternate enable and disable backend
status_JAX_backend() |> expect_false()

enable_JAX_backend()
status_JAX_backend() |> expect_true()

disable_JAX_backend()
status_JAX_backend() |> expect_false()

enable_JAX_backend()
status_JAX_backend() |> expect_true()

disable_JAX_backend(hard_unload = TRUE)
status_JAX_backend() |> expect_false()

enable_JAX_backend()
status_JAX_backend() |> expect_true()
