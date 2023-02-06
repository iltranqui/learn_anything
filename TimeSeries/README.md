# 1st Attempt at TimeSeries

### Defnition: 
A time series is a collection of data points collected at regular intervals of time. It is a sequence of observations of a certain phenomenon over time. For example, stock prices, weather data, and population counts are all examples of time series data.

TimeSeries are composed of 4 main components: 

1. T - Trend   ( Time Variant )
2. J - Jump    ( Time Variant )
3. P - Period  ( Time Variant )
4. S - Stochastics Terms ( Time Invariant )

- Trend: Caused by long term changes in series statistical parameters such as means, 
- Jumps: Edivend short term changes in the dataset, whcih can be determine if significant by tests. 
- Period: Appear in series due to regular and oscillating changes with relatively constant time distances. 
- Shold alsp add outliers detection ( big major step, need to add it ) -> Grubbs Test, Moving Average and Movin Median

## Evaluation and Stationarizary 

1st approach to a time series would be to understand if the selected timne series is under some sort of time variance like a trend, jump or Period. to dtermine such condition there exists particular Statistica Tests for each which can help us determine such condition. 

Time Series are composed by Filling Missing Data

Trend | Jump  | Period | Stocastic | Stationarity
--- | --- | --- | --- | AugmentedDickey-FullerTest
Seconds | 301 | 283 | --- | ---
--- | --- | --- | --- | ---

Table is a WRong appraoch, make a list of all possible Tests. 


| Trend         | Jump     | Stationarity | Period |
|--------------|-----------|------------|----------|
| Mann-Kendal Test | Man-Whitney Test      | *Augmented Dickey-Fuller Test*  | Fisher's Test | 
|     |  | KPSS       |  | 
|       | | Leyboube-McCabe Test       |  | 
|       |  | Correlograms        |  | 

