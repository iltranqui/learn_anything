# Outliers and Missing Data

It is important when you import a time series dataset to obtain a clean dataset where the next steps in time series analysis: 

* Detecting Outliers 
* Filling Missing Data 

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

The test assumes a normal sample distribution for the data and
determined whether the extreme data are outliers or not.



$$ G_{\text {crit. }} \approx \frac{(n-1) t_{\alpha / 2 N,(N-2)}}{\sqrt{N\left(N-2+t_{\alpha / 2 N,(N-2)}^2\right)}} $$ 

```matlab
%Outliers detection
otlr = isoutlier(input,'grubbs');
```

### Generalized extreme studentized deviate test

The generalized extreme studentized deviate (ESDG) test is similar in nature to the Grubbs test and assumes that a univariate data set follows an approximately normal distribution. More accurate than Grubbs test in finding outliers.

$$ R_{\text {crit. }} \approx \frac{(n-i) t_{p, n-i-1}}{\sqrt{(n-i+1)\left(n-i-1+t^2{ }_{p, n-i-1}\right)}}, \quad i=1,2,3 . ., r $$

```matlab
%Outliers detection
otlr = isoutlier(input,'gesd');
```


### Moving average and moving median

Moving average (moving/rolling mean) (MA) and moving median (MM) are two
additional methods of finding outliers in a given time series. The data point is deemed
an outlier if it is more than three local standard deviations from the local average or
three local scaled median absolute deviation (MAD) from the local median over a specific
period (or window length) for moving average and moving median methods, respectively

$$ \begin{gathered}
\text { movmean }_w=\frac{1}{w} \sum_{i=1}^w x_i \\
\text { scaled MAD }=C\left(\text { median }\left(\mid x_i-\text { median } \mid\right)\right), \quad i=1,2,3, \ldots, n \\
C=\frac{1}{\sqrt{2 \operatorname{CEE}(3 / 2)}}
\end{gathered} \\ mormCDF =(\sqrt{2})  \operatorname{ICE}(2p)$$

```matlab
%Outliers detection
otlr = isoutlier(input,'movmean', window);
otlr = isoutlier(input,'movmedian', window);
```

### Quartiles and percentiles

quartiles and percentiles are two methods of detecting outliers. . In the quartiles method, the threshold for outliers is set at 1.5 interquartile ranges above the upper quartile or below the lower quartile. Any data
points outside these thresholds are considered to be outliers.

```matlab
%Outliers detection
otlr = isoutlier(input,'percentiles', [lower upper]);
```

### Replacing the Outliers



```matlab
% Replacing (and finding) outliers with interpolation methods
otlrF = filloutliers(input, 'fillmethod')
%or
otlrF = filloutliers(input, 'fillmethod', 'findmethod')
%or
otlrF = filloutliers(input, 'fillmethod', 'movmethod', window)
%or
otlrF = filloutliers(input, 'fillmethod', 'percentiles', [threshold])
```

# Summary 

| Summary | Function | Finding Method | Replacing Method | 
| -------- | -------- | -------- | ------- | 
| Part 1 | Only detection  |Grubbs | - |
| Part 2  | Only replacing | Makima | - |
| Part 3  | Detection and replacing | clip | gesd |
| Part 4  | Detection and replacing |linear | moving mea |
| Part 5  | Detection and replacing | pchip |percentiles | 
