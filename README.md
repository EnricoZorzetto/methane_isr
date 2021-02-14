# Intermittent surface renewals and Methane hotspots in natural peatlands 

This repository contain python codes to analyze atmospheric turbulence measurements (wind velocity and scalar time series)

The software computes scalar fluxes using eddu covariance, wavelets, and surface renewal theory techniques, with a focus on partitioning fluxes of quantities characterized by intermittent sources at the ground (such as methane over natural wetlands). 

Methane fluxes and Intermittent surface renewals

### Installation

The python code requires Python 3.8 and common scientific packages (Numpy, Scipy, Matplotlib, Pywavelets, and Scikit-Learn). We recomment setting up a conda environment as follows:

```
$ git clone repo /wherever/you/like/
$ cd /wherever/you/like/
$ conda env create --file environment.yaml 
$ conda activate methane
```

### Usage

To replicate the analysis in the paper, run the following codes:

```
$ Methane_preprocessing.py 
$ Methane_analysis.py
$ Meth_plot.py
```


The first script process raw eddy covariance measurements (This step may vary if you use a different dataset), the second performs the analysis, the third produce a number of figures.

### For more information

See the publication, or contact me at enrico dot zorzetto at duke dot edu

