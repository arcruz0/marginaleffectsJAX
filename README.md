# marginaleffectsJAX

A JAX Backend for [`marginaleffects`](https://github.com/vincentarelbundock/marginaleffects/). Under construction!

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
#> JAX is now the backend for `marginaleffects`. Run `disable_JAX_backend()` to disable.
mod <- lm(mpg ~ hp, mtcars)
head(predictions(mod))
#> 
#>  Estimate Std. Error    z Pr(>|z|)     S 2.5 % 97.5 %
#>      22.6      0.777 29.1   <0.001 614.7  21.1   24.1
#>      22.6      0.777 29.1   <0.001 614.7  21.1   24.1
#>      23.8      0.873 27.2   <0.001 539.6  22.0   25.5
#>      22.6      0.777 29.1   <0.001 614.7  21.1   24.1
#>      18.2      0.741 24.5   <0.001 438.7  16.7   19.6
#>      22.9      0.803 28.6   <0.001 594.1  21.4   24.5
#> 
#> Type: response
```

## Supported calls (*very* preliminary)

| Call  | Supported? | Tested? |
| ------------- | - | - |
| `predictions(mod)`  | Yes | No |
| `predictions(mod, by = TRUE)`  | TO-DO | No |
| `predictions(mod, by = variable)`  | TO-DO | No |

