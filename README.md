# plot_utils
Plotting library for conformal prediction metrics, intended to facilitate fast testing in e.g. notebooks.  

## Examples
Example usage can be found in the [User Guide Classification notebook](python/examples/User_guide_classification.ipynb), [User Guide Regression notebook](python/examples/User_guide_regression.ipynb) and [Nonconformist+plot_utils notebook](python/examples/Nonconformist_and_plot_utils.ipynb).

## Package dependencies
See [requirements.txt](python/requirements.txt) for package dependencies used in our development. Here are links to the libraries:

- [matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/) - **Optionally!**

## API

### Data format
The code internally use numpy ndarrays for matrices and vectors, but tries to be agnostic about input being either list, arrays or Pandas equivalents. But for performance reasons it is recommended that conversion to numpy format is done when using several of the methods in this library, as a new conversion would be performed for each function call.

### Rendering backends
Internally this library requires [matplotlib](https://matplotlib.org/) and (optionally) [Seaborn](https://seaborn.pydata.org/). Only the `plot_confusion_matrix_heatmap` has a hard requirement for seaborn to be available, otherwise this library only interacts with the matplotlib classes/functions and use seaborn-settings for generating somewhat nicer plots (in our opinion). Styling and colors can always be changed through the matplotlib API. 

### Data loading 
To simplify loading and conversions of data the `plot_utils` library now has some utility functions for loading CSV files with predictions or validation metrics (typically generated using [CPSign](https://github.com/arosbio/cpsign). For regression we require predictions to be the same as used in [nonconformist](https://github.com/donlnz/nonconformist), using 2D or 3D tensors in numpy ndarrays of shape `(num_examples,2)` or `(num_examples,2,num_significance_levels)`, where the second dimension contains the lower and upper limits of the prediction intervals.


## Supported plots
### Classification
* Calibration plot
* Label ratio plot, showing ratio of single/multi/empty predictions for each significance level
* p-value distribution plot: plot p-values as a scatter plot
* "Bubble plot" confusion matrix
* Heatmap confusion matrix

### Regression 
* Calibration plot
* Efficiency plot (mean or median prediction interval width vs significance)
* Prediction intervals (for a given significance level)

## Set up 
To use this package you clone this repo and add the `<base-path>/python/src/` directory to your `$PYTHONPATH`. 

## Developer notes
We should aim at supplying proper docstrings, following the [numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html).

### Testing
All python-tests are located in the [tests folder](python/tests) and are meant to be run using [pytest](https://docs.pytest.org). Test should be started from standing in the `python` folder and can be run "all at once" (`python -m pytest`), "per file" (`python -m pytest tests/pharmbio/cp/metrics/clf_metrics_test.py`), or a single test function (`python -m pytest tests/pharmbio/cp/metrics/clf_metrics_test.py::TestConfusionMatrix::test_with_custom_labels`). 
- **Note1:** The invocation `python -m pytest [opt args]` is preferred here as the current directory is added to the python path and resolves the application code automatically. Simply running `pytest` requires manual setup of the `PYTHONPATH` instead.
- **Note2:** The plotting tests generate images that are saved in the [test_output](python/tests/test_output) directory and these should be checked manually (no good way of automating plotting-tests).

### TODOs:

Add/finish the following plots:
 - [x] calibration plot - Staffan
 - [x] 'area plot' with label-distributions - Staffan
 - [x] bubbel-plot - Jonathan
 - [x] heatmap - Staffan
 - [x] p0-p1 plot - Staffan
 - [x] Add regression metrics
 - [x] Add plots regression


### Change log:
- **0.1.0**: 
    * Added `pharmbio.cpsign` package with loading functionality for CPSign generated files, loading calibration statistics, efficiency statistics and predictions.
    * Updated plotting functions in order to use pre-computed metrics where applicable (e.g. when computed by CPSign).
    * Added possibility to add a shading for +/- standard deviation where applicable, e.g. calibration curve
    * Updated calibration curve plotting to have a general `plotting.plot_calibration` acting on pre-computed values or for classification using `plotting.plot_calibration_clf` where true labels and p-values can be given.
    * Update parameter order to make it consistent across plotting functions, e.g. ordered as x, y (significance vs error rate) in the plots. 
    * Added a utility function for setting the seaborn theme and context using `plotting.update_plot_settings` which updates the matplotlib global settings. *Note* this will have effect on all other plots generated in the same python session if those rely on matplotlib. 
