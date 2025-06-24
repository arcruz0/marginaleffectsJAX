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
mod <- lm(mpg ~ hp, mtcars)
head(predictions(mod))
```
