# Outliers and Missing Data

It is important when you import a time series dataset to obtain a clean dataset where the next steps in time series analysis: 

* Detecting Outliers 
* filling Missing Data 

## Linear Interpolation

Linear interpolation, sometimes referred to as the weighted average of two points, is the simplest method of interpolation that can be Widely applied through a range of techniques.

You can use 
* Spline method
* Pchip method
* Makia method

## Detecting Outliers

Outliers can be described as the extreme data points that are recorded and markedly
deviate from core statistical measurements of other data points. Outlying data points
may be the result of machine or human errors. 

Outliers are important as they can impact decision-making process and modeling by skewing the results. They can also be used to detect important information 

There are many ways to detect outliers in time series.Grubbs test,generalized extreme
studentized deviate test (ESDG), average and moving average,median and moving median
of data, and interquartile range are a few of the tests that can be easily applied to time
series in the MATLAB environment.

### Grubbs Test

