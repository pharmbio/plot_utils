# plot_utils
Repo that groups utility plotting functions for e.g. plotting of Conformal prediction metrics.

Example usage can be found in the [User Guide notebook](python/User_guide_plotting.ipynb).


## TODOs:

Add/finish the following plots:
 - [x] calibration plot - Staffan
 - [x] 'area plot' with label-distributions - Staffan
 - [x] bubbel-plot - Jonathan
 - [x] heatmap - Staffan
 - [x] p0-p1 plot - Staffan
 - [ ] Add regression metrics
 - [ ] Add plots regression

## Regression 
Have to decide over the API here, the Nonconformist package (https://github.com/donlnz/nonconformist) seems to return 
(n_examples, 2) numpy ndarrays. If several intervals are given, add a third dimension and add a (n_examples,2) for each
new significance level. Are we OK with this setup?
