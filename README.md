# plot_utils
Plotting library for conformal prediction metrics, intended to facilitate fast testing in e.g. notebooks.  

## Examples
Example usage can be found in the [User Guide notebook](python/User_guide_plotting.ipynb).

## Dependencies
- [matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/) - **Optionally!**

## API

### Data format
The code internally use numpy ndarrays for matrices and vectors, but tries to be agnostic about input being either list, arrays or Pandas equivalents. But for performance reasons it is recommended that conversion to numpy format is done when using several of the methods in this library, as a new conversion would be performed for each function call.

### Rendering backends
Internally this library requires [matplotlib](https://matplotlib.org/) and (optionally) [Seaborn](https://seaborn.pydata.org/). Only the `plot_confusion_matrix_heatmap` has a hard requirement for seaborn to be available, otherwise this library only interacts with the matplotlib classes and use the seaborn-settings for generating somewhat nicer plots (in our opinion). Styling and colors can always be changed through the matplotlib API. 

## Supported plots
### Classification
* Calibration plot
* Label ratio plot, showing ratio of single/multi/empty predictions for each significance level
* p-value distribution plot: plot p-values as a scatter plot
* "Bubble plot" confusion matrix
* Heatmap confusion matrix

### Regression 
**TODO**

## Set up 
To use this package you clone this repo and add the `<base-path>/python/src/` directory to your `$PYTHONPATH`. 

## Developer notes
We aim at supplying proper docstrings, following the [numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).

### TODOs:

Add/finish the following plots:
 - [x] calibration plot - Staffan
 - [x] 'area plot' with label-distributions - Staffan
 - [x] bubbel-plot - Jonathan
 - [x] heatmap - Staffan
 - [x] p0-p1 plot - Staffan
 - [ ] Add regression metrics
 - [ ] Add plots regression

_Regression_

Have to decide over the API here, the [Nonconformist package](https://github.com/donlnz/nonconformist) seems to return 
(n_examples, 2) numpy ndarrays (lower and upper bounds for predictions). If several significance levels are used, add a third dimension making up the predictions made for each significance. Are we OK with this convention?
