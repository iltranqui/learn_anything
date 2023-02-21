# Time Series

Noted based on STOCHASTIC MODELINGA Thorough Guide to Evaluate, Pre-Process, Model and Compare Time Series with MATLAB Software Hossein Bonakdari
Mohammad Zeynoddin

The book is divided into: 

1. Preparation and stationarizing
2. Distribution evaluation and normalizing
3. Stochastic modeling
4.  Goodness-of-fit and precision criteria
5. Forecasting time series by deep learning and hybrid methods

### Definition: 
A time series is a collection of data points collected at regular intervals of time. It is a sequence of observations of a certain phenomenon over time. For example, stock prices, weather data, and population counts are all examples of time series data.

TimeSeries are composed of 4 main components: 

<details>
<summary> Time Series Components </summary>

|Concept|Explnation|Details|
|---|---|---|
|T - Trend   ( Time Variant )|Caused by long term changes in series statistical parameters such as means|---|
|J - Jump    ( Time Variant )|Edivend short term changes in the dataset, whcih can be determine if significant by tests. |---|
| P - Period  ( Time Variant )|Appear in series due to regular and oscillating changes with relatively constant time distances. |---|
|S - Stochastics Terms ( Time Invariant )|---|---|
</details>


## Evaluation and Stationarizary 

1st approach to a time series would be to understand if the selected timne series is under some sort of time variance like a trend, jump or Period. to dtermine such condition there exists particular Statistica Tests for each which can help us determine such condition. 

Time Series are composed by Filling Missing Data


| Trend         | Jump     | Stationarity | Period |
|--------------|-----------|------------|----------|
| Mann-Kendal Test | Man-Whitney Test      | *Augmented Dickey-Fuller Test*  | Fisher's Test | 
|     |  | KPSS       |  | 
|       | | Leyboube-McCabe Test       |  | 
|       |  | Correlograms        |  | 

# Stationary Tests

Here is a list of statistical tests that can be used to determine if a time series is stationary or not:

Augmented Dickey-Fuller (ADF) Test: This is a widely used test for stationarity, which tests the null hypothesis that a time series has a unit root (i.e., non-stationary). The test statistic is compared to critical values from the ADF distribution to determine stationarity.

Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test: This test is also used to test for stationarity, but it has a different null hypothesis. The KPSS test tests the hypothesis that a time series is trend-stationary (i.e., has a constant mean).

Phillips-Perron (PP) Test: This test is an improved version of the ADF test that accounts for autocorrelation in the residuals.

Cumulative Sum (CUSUM) Test: This is a graphical test that plots the cumulative sum of the residuals over time. If the time series is stationary, the plot should oscillate around zero, without trending up or down.

Hurst Exponent Test: This test measures the long-term memory of a time series. If the Hurst exponent is close to 0.5, the time series is considered to be stationary. If it is greater than 0.5, the time series has positive autocorrelation (i.e., it is trending upwards) and is considered to be non-stationary. If it is less than 0.5, the time series has negative autocorrelation (i.e., it is trending downwards) and is considered to be non-stationary.

These tests can provide useful information about the stationarity of a time series, but they should not be relied upon solely. Other methods, such as visual inspection of the time series plot, should also be used to assess stationarity.

# Jump Tests

Here is a list of statistical tests that can be used to determine if a time series has a jump or not:

Lee-Myers Test: This test is based on the difference between a time series and its moving average. Large differences are indicative of jumps in the time series.

Variance Ratio Test: This test is based on the ratio of the variance of the differences between two time series with different time horizons. Large ratios are indicative of jumps in the time series.

GARCH-Jump Test: This test is based on a GARCH (Generalized Autoregressive Conditional Heteroscedasticity) model, which is a time series model that accounts for both autocorrelation and volatility. The test uses the residuals from the GARCH model to detect jumps in the time series.

Jump Detection Algorithm: This is a computational method that detects jumps in a time series by iteratively dividing the time series into smaller segments and comparing the mean and variance of each segment.

Changes in the Mean Test: This test is based on the differences between the mean of the time series and its moving average. Large differences are indicative of jumps in the time series.

These tests can provide useful information about the presence of jumps in a time series, but they should not be relied upon solely. Other methods, such as visual inspection of the time series plot, should also be used to assess the presence of jumps.

# Seasonlaity Test

Here is a list of statistical tests that can be used to determine if a time series has seasonality or not:

Augmented Dickey-Fuller with Seasonal Difference Test (ADF-SD Test): This test is an extension of the Augmented Dickey-Fuller (ADF) Test, which accounts for seasonality in the time series. The test statistic is compared to critical values from the ADF-SD distribution to determine the presence of seasonality.

Seasonality Tests based on Autocorrelation Function (ACF): This test involves plotting the autocorrelation function (ACF) of the time series and visually examining the plot for peaks at multiples of the seasonal frequency.

Seasonality Tests based on Partial Autocorrelation Function (PACF): This test involves plotting the partial autocorrelation function (PACF) of the time series and visually examining the plot for peaks at multiples of the seasonal frequency.

Seasonal Mann-Kendall Test: This test is a non-parametric method for detecting seasonality in a time series. The test statistic is compared to critical values from the Mann-Kendall distribution to determine the presence of seasonality.

Seasonality Tests based on Time-series Decomposition: This test involves decomposing the time series into its trend, seasonality, and residual components using methods such as the STL (Seasonal and Trend decomposition using Loess) or the X-13ARIMA-SEATS method. The presence of a strong seasonal component indicates the presence of seasonality in the time series.

These tests can provide useful information about the presence of seasonality in a time series, but they should not be relied upon solely. Other methods, such as visual inspection of the time series plot, should also be used to assess the presence of seasonality.

# Period Test

Here are some statistical tests that can be used to determine if a time series has a period:

Augmented Dickey-Fuller (ADF) test: This test can be used to determine if a time series has a unit root, which is an indication of non-stationarity. If the series is non-stationary, it may have a repeating pattern, which is an indication of a period.

Ljung-Box test: This test checks for autocorrelation in the residuals of a time series model. If the residuals have significant autocorrelation at regular intervals, this suggests the presence of a period.

Spectral analysis: This method involves plotting the power spectral density (PSD) of the time series. If the PSD plot reveals peaks at specific frequencies, this suggests the presence of a period in the time series.

Seasonal Decomposition of Time Series (STL) method: This method involves decomposing the time series into its seasonal, trend, and residual components. If the seasonal component shows repeating patterns, this suggests the presence of a period in the time series.

Autocorrelation Function (ACF) plot: The ACF plot shows the correlation between the time series and lagged versions of itself. If the plot shows significant autocorrelation at specific lags, this suggests the presence of a period in the time series.

# Outliers Test

Here are some statistical tests that can be used to determine if a time series has outliers:

Z-score test: This test calculates the Z-score of each observation in the time series and checks if any observation has a Z-score that is significantly different from zero. Observations with large positive or negative Z-scores are considered outliers.

Grubbs' test: This test calculates a test statistic based on the difference between the mean and median of the time series, as well as the standard deviation. If the test statistic is larger than a critical value, the observation is considered an outlier.

Dixon's test: This test involves dividing the range of the time series into quarters, and checking if any observation falls outside the range of the nearest quartile. If an observation falls outside the range, it is considered an outlier.

Box plot: A box plot displays the minimum, first quartile, median, third quartile, and maximum of a time series. Observations that fall outside the range of the minimum or maximum are considered outliers.

Tukey's method: This method involves calculating the interquartile range (IQR) and considering any observations that fall outside the range of 1.5 times the IQR as outliers.
