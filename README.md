# marginaleffectsJAX

A JAX backend for [`marginaleffects`](https://github.com/vincentarelbundock/marginaleffects/). Under construction!

## Installation

``` r
install.packages("remotes") # if `remotes` is not installed
remotes::install_github("arcruz0/marginaleffectsJAX")
```
## Usage (*very* preliminary)
``` r
library(marginaleffects)
library(marginaleffectsJAX)
enable_JAX_backend()
#> JAX is now a backend for `marginaleffects`. Run `disable_JAX_backend()` to disable.

mod <- lm(mpg ~ hp + am, mtcars)

predictions(mod) |> head()
#> 
#>  Estimate Std. Error    z Pr(>|z|)     S 2.5 % 97.5 %
#>      25.4      0.818 31.0   <0.001 700.5  23.8   27.0
#>      25.4      0.818 31.0   <0.001 700.5  23.8   27.0
#>      26.4      0.850 31.1   <0.001 701.1  24.7   28.1
#>      20.1      0.775 25.9   <0.001 490.0  18.6   21.6
#>      16.3      0.677 24.0   <0.001 421.6  15.0   17.6
#>      20.4      0.796 25.6   <0.001 478.6  18.8   22.0
#> 
#> Type: response

predictions(mod, by = TRUE)
#> 
#>  Estimate Std. Error    z Pr(>|z|)   S 2.5 % 97.5 %
#>      20.1      0.514 39.1   <0.001 Inf  19.1   21.1
#> 
#> Type: response

predictions(mod, by = "am")
#> 
#>  am Estimate Std. Error    z Pr(>|z|)     S 2.5 % 97.5 %
#>   0     17.1      0.667 25.7   <0.001 481.2  15.8   18.5
#>   1     24.4      0.807 30.2   <0.001 664.5  22.8   26.0
#> 
#> Type: response
```

## Supported `marginaleffects` calls (*very* preliminary)

(Only models of class `lm` are supported).

| Call  | Supported? |
| :--- |   :---:    |
| `predictions(mod)`  | ✅ | 
| `predictions(mod, by = TRUE)` <br> `avg_predictions(mod)`  | ✅  | 
| `predictions(mod, by = "var")` <br> `plot_predictions(mod, by = "var")`  | ✅ |
| `predictions(mod, by = "var", wt = "wvar")`  | 🔜 |
| `predictions(mod, by = ..., byfun = sum)`  | 🔜 |
| `comparisons(mod, ...)`  | 🔜 |



