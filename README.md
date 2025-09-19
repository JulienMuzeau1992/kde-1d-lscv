# kde-1d-lscv
Univariate Kernel Density Estimation (KDE) with Least Squares Cross-Validation (LSCV), also known as Unbiased Cross-Validation (UCV).

## Useful links about KDE
- [https://numxl.com/blogs/kde-optimization-unbiased-cross-validation/](https://numxl.com/blogs/kde-optimization-unbiased-cross-validation/)
- [https://bookdown.org/egarpor/NP-UC3M/kde-i.html](https://bookdown.org/egarpor/NP-UC3M/kde-i.html)

## Requirements
- numpy (2.2.6)
- scikit-learn (1.7.2)
- joblib (1.5.2)
- tqdm (4.67.1)
- matplotlib (3.10.6)

## Todo
- Other kernels (Gaussian only for now)
- Biased and Maximum Likelihood Cross-Validation (BCV & MLCV)
- Scott and Silverman plug-in methods
- Sheather & Jones method, also known as Direct Plug-In (DPI)
- Optimisation instead of grid search
- Separate dataset into train and test