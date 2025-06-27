library(microbenchmark)
library(data.table); setDTthreads(1)
library(marginaleffects) 
library(marginaleffectsJAX)
c("marginaleffects", "marginaleffectsJAX") |> lapply(packageVersion)
#> [[1]]
#> [1] '0.27.0.8'
#> 
#> [[2]]
#> [1] '0.0.1'
enable_JAX_backend()

f_bench <- function(N, K, times_each = 10){
  # create dataset with K indep. vars, half std. normal, half dummies
  d <- data.table(y = rnorm(N))
  for (k in seq_len(K)){
    varname <- paste("x", k, sep = "")
    if (k <= K / 2) {
      set(d, j = varname, value = rnorm(N))
    } else {
      set(d, j = varname, value = rbinom(N, size = 1, prob = .5))
    }
  }
  
  # estimate model (w/o intercepts)
  mod <- lm(y ~ . - 1, d)
  
  # benchmark predictions()
  bench_gen <- microbenchmark(
    no_jax = {disable_JAX_backend(); predictions(mod)},
    jax = {enable_JAX_backend(); predictions(mod)},
    times = times_each
  )
  
  # benchmark predictions(by = T)
  bench_byT <- microbenchmark(
    no_jax = {disable_JAX_backend(); predictions(mod, by = T)},
    jax = {enable_JAX_backend(); predictions(mod, by = T)},
    times = times_each
  )
  
  # benchmark predictions(by = ) with one of the variables
  bench_by <- microbenchmark(
    no_jax = {disable_JAX_backend(); predictions(mod, by = tail(colnames(d), 1))},
    jax = {enable_JAX_backend(); predictions(mod, by = tail(colnames(d), 1))},
    times = times_each
  )
  
  # combine benchmarks and take median and max time
  bench <- rbind(
    as.data.table(bench_gen)[, bench := "gen"],
    as.data.table(bench_byT)[, bench := "byT"],
    as.data.table(bench_by)[, bench := "by"]
  )[
    , .(median_time_ms = median(time / 1e6),
        max_time_ms = max(time / 1e6)), 
    by = .(expr, bench)
  ][
    , ":="(N = N, K = K)
  ]
  
  return(bench[])
}

# execute branch over range of values of N and K
hyp <- expand.grid(N = c(1e4, 1e5, 1e6), K = c(10, 50, 100))

l_results <- lapply(
  seq_len(nrow(hyp)), 
  \(i) {
    message(sprintf("Benchmark %s: N=%s, K=%s", i, hyp$N[i], hyp$K[i]))
    suppressMessages(f_bench(N = hyp$N[i], K = hyp$K[i]))
  }
)

results <- rbindlist(l_results)
fwrite(results, "benchmarks/benchmark_predictions.csv")

# plot results
library(ggplot2)
library(patchwork)

labeller_N <- labeller(
  N = \(x) {
    num <- formatC(as.numeric(x), format = "f", big.mark = ",", digits = 0)
    return(sprintf("N=%s", num))
  }
)

f_plot <- \(y_var, y_label){
  p1 <- ggplot(results[bench == "gen"], aes(x = K, y = get(y_var) / 1000, color = expr)) +
    geom_point() +
    geom_line() +
    scale_x_continuous(breaks = hyp$K) +
    facet_wrap(~N, labeller = labeller_N, scales = "free") +
    labs(y = "", subtitle = "predictions()", color = "") +
    theme(panel.grid.minor = element_blank())
  
  p2 <- ggplot(results[bench == "byT"], aes(x = K, y = get(y_var) / 1000, color = expr)) +
    geom_point() +
    geom_line() +
    scale_x_continuous(breaks = hyp$K) +
    facet_wrap(~N, labeller = labeller_N, scales = "free") +
    labs(y = y_label, subtitle = "predictions(by = T)", color = "") +
    theme(panel.grid.minor = element_blank())
  
  p3 <- ggplot(results[bench == "by"], aes(x = K, y = get(y_var) / 1000, color = expr)) +
    geom_point() +
    geom_line() +
    scale_x_continuous(breaks = hyp$K) +
    facet_wrap(~N, labeller = labeller_N, scales = "free") +
    labs(y = "", subtitle = 'predictions(by = "var")', color = "") +
    theme(panel.grid.minor = element_blank())
  
  
  p1 / p2 / p3
}

p_median <- f_plot("median_time_ms", "Median time (in seconds)")
p_max <- f_plot("max_time_ms", "Max time (in seconds)")

ggsave(plot = p_median, filename = "benchmarks/benchmark_predictions_p_median.png",
       width = 7, height = 5)

ggsave(plot = p_max, filename = "benchmarks/benchmark_predictions_p_max.png",
       width = 7, height = 5)
