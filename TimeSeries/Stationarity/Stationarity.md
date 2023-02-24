# Stationarity 

> if i understood correctly, 1st you have to stationarize your dataset and then you have to normalize it -> only then you can apply your stochastic modeling. 

In simple terms, stationarity can be described as a flat looking time series – one where the
data points fluctuate around a horizontal line and the measures of central tendency and
variability (mean and variance, respectively) do not change significantly over time. The expectation is that the time series is independent over time. 

stationarity in time series can be classified as either weak or strong. In a strong
stationary time series, not only the mean and variance are constant in time but also the
covariance of the data relies merely on the time lag. A weak stationary
time series, however, only has a constant mean and it does not alter from surrounding
variables.

> This state of stationarity is widely used in modeling, particularly in stochastic modeling 

Conversely, a random walk is a time series with time correlation with previous lag $ Xt = X_{t−1} + ε_{t} :  ε_{t}$ white noise).

The vast majority of natural time series, however, are nonstationary and
constitute an array of sophisticated structures. Therefore, preprocessing and corrections
prior to modeling is a perquisite step.

### Unit roots

Unit root tests are data-generating processes with this assumption that data
are random walk and by first-order differencing, a stationary process is obtained  $ Xt = X_{t−1} + ε_{t} $ and $ Xt = ξ $  stationary process. 

## Augmented Dickey–Fuller test

In this test, stationarity is not directly determined
but based on the existence or lack of the unit root, the stationarity can be evaluated.
This analysis fits a linear and the second-order model, then, if there is a unit root for the
model, the series is nonstationary, and in the absence of the unit root, the series will be
stationary

$ \begin{gathered}
\Delta x(t)=\alpha+\beta_1 t+\beta_2 t^2+\gamma x(t-1)+\varphi_1 \Delta x(t-1)+\ldots+\varphi_{p-1} \Delta x(t-p+1)+\varepsilon(t) \\
\tau=\frac{\hat{\gamma}}{\sigma_{\hat{\gamma}}} \\
\left\{\begin{array}{l}
\mathrm{H}_0: \gamma=0 \\
\mathrm{H}_1: \gamma<0
\end{array}\right.
\end{gathered} $

$\Delta$ is the first-order differential operator; α, β1, and β2 are a
constant, linear trend coefficients, and second-order (respectively) that can be considered
zero; $\varphi$ is the i_th correlation coefficient; p the maximum degree of auto-regression; $\tau$
the Augmented Dickey–Fuller (ADF) test statistic;

* H0 is zero assumption based on unit root -> nonstationarity
* H1 is the alternative assumption that there is no unit root -> stationarity

```matlab
% stationarity, ADF test
[adf_H, adf_pval, adf_stat, adf_crit] = adftest(input);
```
H = 1 -> nonstationtity 
H = 0 -> stationarity

## KPSS 

the KPSS test can be used to evaluate stationarity in the presence of a deterministic trend. The null
hypothesis may not be rejected even if there is an increasing or decreasing trend.

$$
\begin{gathered}
S^2(l)=\frac{1}{n} \sum_{t=1}^n e_t^2+\frac{2}{n} \sum_{s=1}^l w(s, l) \frac{1}{n} \sum_{t=s 1}^n e_t e_{t-s} \\
w(s, l)=1-s(l+1)
\end{gathered}
$$
age square of errors between time 1 and $t$. Th vel or for trends detection are as follows:
$$
\begin{aligned}
& \eta_\mu=\frac{1}{n^2} \sum_{t=1}^n \frac{S_t^2}{S^2(l)} \\
& \eta_\tau=\frac{1}{n^2} \sum_{t=1}^n \frac{S_t^2}{S^2(l)}
\end{aligned}
$$

The difference between the two relationships can be attributed to the magnitude of
the residuals.

```matlab
%stationarity, KPSS test
[kpss_H, kpss_pVal, kpss_stat, kpss_crit] = kpsstest(input)
```

* H = 1 indicates rejection of the trend-stationary null in favor of the unit root
alternative. -> nonstationarity
* H = 0 indicates failure to reject the trend-stationary null -> stationarity

### PP-test - Phillips–Perron test

The Phillips–Perron (PP) test is a generalized form of Dickey–Fuller test. Similar to
the Dickey–Fuller test, the PP test develops a regression line to assess the stationarity

$ X_{t} = c + δ_{t} + α_{t}X_{t-1} + ε_{t} $ 

where $c$ and $δ$ are drift and deterministic trend coefficients and $ε_{t}$ is the error term, which is expected to have zero mean. the PP test applies
a nonparametric correction to the test statistic to assess the impact of correlations on
the adjustment residuals. 

```matlab
% stationarity, PP test
[pp_H, pp_pVal, pp_stat, pp_crit] = pptest(input)
```
* Output: 
 H: the logical value of the test decisionH;
pval: P-values of the statistic, pval; 
 stat: statistic itself,
crit: critical values of the test,

* H = 1 -> the series is stationary
* H = 0 indicates rejection of the unit-root null or there is a unit root for the series  -> The series is nonstationary.

# Stationarizing methods

After testing time series for deterministic terms and detecting nonstationary factors,they should be stationarized using appropriate methods. 

